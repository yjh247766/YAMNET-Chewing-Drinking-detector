import 'dart:math' as math;
import 'dart:typed_data';

/// 목넘김(꿀꺽) 소리 감지기 — 순수 Dart, 라이브러리 추가 없음
///
/// 실측 기반 설계 (Galaxy S20+, 2026-05-15):
///
///  소리      | 배율    | 중/저  | 고/저  | 선행고에너지
///  목넘김    | 10.8x  | 0.474 | 0.031 | 없음
///  의자끌기  | 3~103x | 0.33~1.38 | 0.04~0.65 | 항상 있음
///  키보드    | 3~6x   | 0.40~1.86 | 0.17~1.17 | 가끔 있음
///
/// 3단계 필터:
///   1. 고역/저역 비율 ≥ 0.45 → 기침으로 차단 + 500ms 쿨다운
///   2. 중역/저역 비율 > 0.50 → 말소리/키보드 차단 (중역이 강하면 목넘김 아님)
///   3. 직전 500ms(5프레임) 내 배율 ≥ 3.0 구간 존재 → 의자/충격음 차단
///      (목넘김은 조용한 배경에서 단발로 발생)
class DrinkDetector {
  static const int sampleRate   = 16000;
  static const int subChunkSize = 1600; // 100ms

  // ── 저역 필터 계수 (100~500Hz) ───────────────────
  static const _hpB = [0.9691, -1.9382, 0.9691];
  static const _hpA = [1.0000, -1.9380, 0.9384];
  static const _lpB = [0.00038, 0.00076, 0.00038];
  static const _lpA = [1.0000,  -1.9645, 0.9657];

  // ── 중역 필터 계수 (500~3000Hz) ──────────────────
  static const _hp5B = [0.9565, -1.9130, 0.9565];
  static const _hp5A = [1.0000, -1.9112, 0.9150];
  static const _lp3kB = [0.0340, 0.0680, 0.0340];
  static const _lp3kA = [1.0000, -1.4218, 0.5178];

  // ── 고역 필터 계수 (3000~7000Hz) ─────────────────
  static const _hp3kB = [0.7265, -1.4529, 0.7265];
  static const _hp3kA = [1.0000, -0.9979, 0.2176];
  static const _lp7kB = [0.5681,  1.1362, 0.5681];
  static const _lp7kA = [1.0000,  0.9073, 0.3575];

  // ── 파라미터 (실측 기반) ──────────────────────────
  static const double _minAbsThreshold   = 0.0003; // 노이즈 플로어
  static const double _riseMultiplier    = 3.0;    // 배경 대비 배율
  static const double _midRatioMax       = 0.50;   // 중/저 비율 상한 (목넘김=0.47, 의자=0.76+)
  static const double _coughHighRatio    = 0.45;   // 고/저 비율 — 기침 판정
  static const int    _coughCooldown     = 5;      // 기침 후 차단 프레임
  static const int    _preCheckFrames    = 5;      // 선행 고에너지 탐지 윈도우 (500ms)
  static const double _preCheckRise      = 3.0;    // 선행 고에너지 판정 배율
  static const int    _historyFrames     = 30;     // 배경 추정 윈도우
  static const int    _maxActiveFrames   = 10;     // 최대 연속 활성 (1초 초과시 노이즈)

  // ── IIR 필터 상태 ────────────────────────────────
  final _hpZ   = [0.0, 0.0];
  final _lpZ   = [0.0, 0.0];
  final _hp5Z  = [0.0, 0.0];
  final _lp3kZ = [0.0, 0.0];
  final _hp3kZ = [0.0, 0.0];
  final _lp7kZ = [0.0, 0.0];

  // ── 히스토리 ─────────────────────────────────────
  final List<double> _bandHistory = [];   // 배경 추정용
  final List<double> _riseHistory = [];   // 선행 에너지 체크용

  // ── 이벤트 상태 ──────────────────────────────────
  int    _activeFrames  = 0;
  bool   _alreadyFired  = false;
  int    _coughCooldownCount = 0;

  // ── 디버그용 ─────────────────────────────────────
  double lastBandRms    = 0.0;
  double lastMidRms     = 0.0;
  double lastHighRms    = 0.0;
  double lastBackground = 0.0;
  double lastRiseRatio  = 0.0;
  double lastMidRatio   = 0.0;
  double lastHighRatio  = 0.0;
  bool   lastIsCough    = false;
  bool   lastInCooldown = false;
  bool   lastPreBlocked = false;
  int    lastActiveFrames = 0;

  DrinkEvent process(Float32List samples) {
    DrinkEvent? result;
    for (int offset = 0; offset + subChunkSize <= samples.length; offset += subChunkSize) {
      final sub   = Float32List.sublistView(samples, offset, offset + subChunkSize);
      final event = _processSub(sub);
      if (event.detected) result = event;
    }
    return result ?? DrinkEvent(
      detected: false,
      bandRms: lastBandRms, midRms: lastMidRms, highRms: lastHighRms,
      background: lastBackground, riseRatio: lastRiseRatio,
      midRatio: lastMidRatio, highRatio: lastHighRatio,
      isCough: lastIsCough, inCooldown: lastInCooldown,
      preBlocked: lastPreBlocked, activeFrames: lastActiveFrames,
    );
  }

  DrinkEvent _processSub(Float32List sub) {
    final lowFilt  = _applyBandpass(sub, _hpZ,   _lpZ,   _hpB,   _hpA,   _lpB,   _lpA);
    final midFilt  = _applyBandpass(sub, _hp5Z,  _lp3kZ, _hp5B,  _hp5A,  _lp3kB, _lp3kA);
    final highFilt = _applyBandpass(sub, _hp3kZ, _lp7kZ, _hp3kB, _hp3kA, _lp7kB, _lp7kA);

    final bandRms  = _rms(lowFilt);
    final midRms   = _rms(midFilt);
    final highRms  = _rms(highFilt);

    final midRatio  = bandRms > 0.0001 ? midRms  / bandRms : 0.0;
    final highRatio = bandRms > 0.0001 ? highRms / bandRms : 0.0;

    // 배경 추정
    _bandHistory.add(bandRms);
    if (_bandHistory.length > _historyFrames) _bandHistory.removeAt(0);
    final background = _backgroundEnergy(_bandHistory);
    final riseRatio  = background > 0 ? bandRms / background : 0.0;

    // 선행 고에너지 히스토리 업데이트
    _riseHistory.add(riseRatio);
    if (_riseHistory.length > _preCheckFrames + 1) _riseHistory.removeAt(0);

    // ── 필터 1: 기침 판정 ────────────────────────
    final isCough = highRatio >= _coughHighRatio;
    if (isCough) {
      _coughCooldownCount = _coughCooldown;
      _activeFrames = 0; _alreadyFired = false;
    } else if (_coughCooldownCount > 0) {
      _coughCooldownCount--;
    }
    final inCooldown = _coughCooldownCount > 0;

    // ── 필터 2: 중역 비율 (말소리/키보드) ──────────
    final midBlocked = midRatio > _midRatioMax;

    // ── 필터 3: 선행 고에너지 (의자/충격음) ─────────
    bool preBlocked = false;
    if (_riseHistory.length >= _preCheckFrames) {
      final preWindow = _riseHistory.sublist(0, _riseHistory.length - 1);
      preBlocked = preWindow.any((r) => r >= _preCheckRise);
    }

    final isActive = bandRms > _minAbsThreshold &&
        riseRatio >= _riseMultiplier &&
        !midBlocked &&
        !isCough &&
        !inCooldown &&
        !preBlocked;

    bool detected = false;
    if (isActive) {
      _activeFrames++;
      if (_activeFrames == 1 && !_alreadyFired) {
        detected = true;
        _alreadyFired = true;
      }
      if (_activeFrames >= _maxActiveFrames) {
        _activeFrames = 0; _alreadyFired = false;
      }
    } else if (!isCough && !inCooldown) {
      _activeFrames = 0; _alreadyFired = false;
    }

    lastBandRms     = bandRms;
    lastMidRms      = midRms;
    lastHighRms     = highRms;
    lastBackground  = background;
    lastRiseRatio   = riseRatio;
    lastMidRatio    = midRatio;
    lastHighRatio   = highRatio;
    lastIsCough     = isCough;
    lastInCooldown  = inCooldown;
    lastPreBlocked  = preBlocked;
    lastActiveFrames = _activeFrames;

    return DrinkEvent(
      detected: detected,
      bandRms: bandRms, midRms: midRms, highRms: highRms,
      background: background, riseRatio: riseRatio,
      midRatio: midRatio, highRatio: highRatio,
      isCough: isCough, inCooldown: inCooldown,
      preBlocked: preBlocked, activeFrames: _activeFrames,
    );
  }

  Float32List _applyBandpass(
    Float32List input,
    List<double> hpZ, List<double> lpZ,
    List<double> hpB, List<double> hpA,
    List<double> lpB, List<double> lpA,
  ) {
    final n   = input.length;
    final hp  = Float32List(n);
    final out = Float32List(n);
    for (int i = 0; i < n; i++) {
      final x = input[i];
      final y = hpB[0] * x + hpZ[0];
      hpZ[0] = hpB[1] * x - hpA[1] * y + hpZ[1];
      hpZ[1] = hpB[2] * x - hpA[2] * y;
      hp[i] = y;
    }
    for (int i = 0; i < n; i++) {
      final x = hp[i];
      final y = lpB[0] * x + lpZ[0];
      lpZ[0] = lpB[1] * x - lpA[1] * y + lpZ[1];
      lpZ[1] = lpB[2] * x - lpA[2] * y;
      out[i] = y;
    }
    return out;
  }

  double _rms(Float32List s) {
    double sum = 0.0;
    for (final v in s) sum += v * v;
    return math.sqrt(sum / s.length);
  }

  double _backgroundEnergy(List<double> history) {
    if (history.isEmpty) return 0.0;
    final sorted = List<double>.from(history)..sort();
    final count  = math.max(1, (sorted.length * 0.4).floor());
    double sum   = 0;
    for (int i = 0; i < count; i++) sum += sorted[i];
    return sum / count;
  }

  void reset() {
    for (final z in [_hpZ, _lpZ, _hp5Z, _lp3kZ, _hp3kZ, _lp7kZ]) {
      z[0] = z[1] = 0.0;
    }
    _bandHistory.clear();
    _riseHistory.clear();
    _activeFrames = 0;
    _alreadyFired = false;
    _coughCooldownCount = 0;
  }
}

class DrinkEvent {
  final bool   detected;
  final double bandRms;
  final double midRms;
  final double highRms;
  final double background;
  final double riseRatio;
  final double midRatio;
  final double highRatio;
  final bool   isCough;
  final bool   inCooldown;
  final bool   preBlocked;   // 선행 고에너지로 차단됨 (의자/충격음)
  final int    activeFrames;

  const DrinkEvent({
    required this.detected,
    required this.bandRms,
    required this.midRms,
    required this.highRms,
    required this.background,
    required this.riseRatio,
    required this.midRatio,
    required this.highRatio,
    required this.isCough,
    required this.inCooldown,
    required this.preBlocked,
    required this.activeFrames,
  });
}
