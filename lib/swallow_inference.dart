/// 목넘김(삼킴) 소리 추론 모듈
/// 사용법: SwallowDetector를 초기화하고 오디오 청크를 feedPcm16으로 전달
/// 
/// 학습 파라미터 (train_swallow.py와 동일하게 맞춰야 함):
///   - 입력: 16kHz, PCM16 mono
///   - 패치: 96 프레임 × 64 mel-bin (약 960ms)
///   - 정규화: (logmel - mean) / std  (swallow_norm_stats.npy 참조)
///   - 임계값: 0.26 (swallow_norm_stats.npy[2])

import 'dart:math';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';

class SwallowDetector {
  static const int kSampleRate    = 16000;
  static const int kNFft          = 400;    // 25ms
  static const int kHopLen        = 160;    // 10ms
  static const int kNMels         = 64;
  static const double kFMin       = 125.0;
  static const double kFMax       = 7500.0;
  static const int kPatchFrames   = 96;     // ~960ms

  // swallow_norm_stats.npy에서 읽은 값
  static const double kNormMean   = -49.6514;
  static const double kNormStd    = 25.1358;
  static const double kThreshold  = 0.06;

  late Interpreter _interp;
  final List<double> _audioBuffer = [];

  /// 초기화: assets/swallow_classifier.tflite 로드
  Future<void> init(String modelPath) async {
    _interp = await Interpreter.fromAsset(modelPath);
  }

  /// PCM16 바이트 스트림을 받아 float32 [-1,1]로 변환 후 버퍼에 추가
  /// 반환값: 목넘김 감지되면 true, 아직 판단 안 됨이면 null
  bool? feedPcm16(Uint8List pcmBytes) {
    final samples = Float32List(pcmBytes.length ~/ 2);
    for (int i = 0; i < samples.length; i++) {
      final lo = pcmBytes[i * 2];
      final hi = pcmBytes[i * 2 + 1];
      int s16 = (hi << 8) | lo;
      if (s16 >= 0x8000) s16 -= 0x10000;
      samples[i] = s16 / 32768.0;
    }
    _audioBuffer.addAll(samples);

    // 패치 1개 만들 수 있을 때만 추론
    final needed = kNFft + (kPatchFrames - 1) * kHopLen; // 15760 samples
    if (_audioBuffer.length < needed) return null;

    final waveform = Float32List.fromList(_audioBuffer.take(needed).toList());
    _audioBuffer.removeRange(0, kHopLen * (kPatchFrames ~/ 2)); // 50% 겹침 슬라이딩

    final patch = _computeLogMelPatch(waveform);
    return _infer(patch);
  }

  /// log-mel spectrogram → (kPatchFrames, kNMels) 패치 계산
  List<List<double>> _computeLogMelPatch(Float32List wav) {
    // Mel filterbank 행렬 (미리 계산된 근사값 사용)
    // 실제 배포 시 precomputed mel_fb.npy를 assets에 포함하거나
    // librosa와 동일한 방식으로 구현
    final frames = <List<double>>[];

    for (int t = 0; t < kPatchFrames; t++) {
      final start = t * kHopLen;
      final frame = Float32List(kNFft);
      for (int i = 0; i < kNFft && (start + i) < wav.length; i++) {
        // Hann window
        final w = 0.5 - 0.5 * cos(2 * pi * i / (kNFft - 1));
        frame[i] = wav[start + i] * w;
      }

      // FFT magnitude
      final mag = _fftMagnitude(frame);

      // Mel filterbank (간략화된 삼각 필터)
      final melEnergies = _applyMelFilterbank(mag);

      // power → dB (ref = max)
      final logMel = melEnergies.map((e) => 10.0 * log(max(e, 1e-10)) / ln10).toList();
      frames.add(logMel);
    }

    // 최대값 기준 dB 정규화 (librosa power_to_db(ref=np.max) 방식)
    double maxDb = frames.expand((f) => f).reduce(max);
    for (final f in frames) {
      for (int i = 0; i < f.length; i++) f[i] -= maxDb;
    }
    return frames;
  }

  /// 입력 패치를 TFLite 모델에 전달 → 목넘김 여부 반환
  bool _infer(List<List<double>> patch) {
    // 입력 shape: [1, 96, 64, 1]
    final input = List.generate(1, (_) =>
      List.generate(kPatchFrames, (t) =>
        List.generate(kNMels, (m) =>
          [(patch[t][m] - kNormMean) / kNormStd])));

    final output = List.generate(1, (_) => [0.0]);
    _interp.run(input, output);
    return output[0][0] >= kThreshold;
  }

  void dispose() => _interp.close();

  // ── 내부 헬퍼 ──────────────────────────────────────────────

  /// 간단한 DFT magnitude (실수 입력)
  /// 실제 배포에서는 FFT 라이브러리 사용 권장 (dart_fft 등)
  List<double> _fftMagnitude(Float32List frame) {
    final n = frame.length;
    final half = n ~/ 2 + 1;
    final mag = List<double>.filled(half, 0.0);
    for (int k = 0; k < half; k++) {
      double re = 0, im = 0;
      for (int n_ = 0; n_ < n; n_++) {
        final angle = -2 * pi * k * n_ / n;
        re += frame[n_] * cos(angle);
        im += frame[n_] * sin(angle);
      }
      mag[k] = sqrt(re * re + im * im);
    }
    return mag;
  }

  /// 삼각 mel filterbank (kNMels개, kFMin~kFMax)
  List<double> _applyMelFilterbank(List<double> mag) {
    final freqBin = (i) => i * kSampleRate / kNFft;
    final hzToMel = (hz) => 2595.0 * log(1.0 + hz / 700.0) / ln10;
    final melToHz = (mel) => 700.0 * (pow(10.0, mel / 2595.0) - 1.0);

    final melMin = hzToMel(kFMin);
    final melMax = hzToMel(kFMax);
    final melPoints = List.generate(kNMels + 2,
      (i) => melToHz(melMin + i * (melMax - melMin) / (kNMels + 1)));

    final energies = List<double>.filled(kNMels, 0.0);
    for (int m = 0; m < kNMels; m++) {
      final fLow  = melPoints[m];
      final fCen  = melPoints[m + 1];
      final fHigh = melPoints[m + 2];
      for (int k = 0; k < mag.length; k++) {
        final f = freqBin(k);
        double w = 0;
        if (f >= fLow  && f <= fCen)  w = (f - fLow)  / (fCen  - fLow);
        if (f >= fCen  && f <= fHigh) w = (fHigh - f)  / (fHigh - fCen);
        energies[m] += w * mag[k] * mag[k]; // power
      }
    }
    return energies;
  }
}
