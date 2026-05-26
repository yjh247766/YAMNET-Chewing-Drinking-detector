import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// YAMNet 모델로 오디오를 521개 클래스로 분류하는 래퍼
///
/// 입력: 16kHz mono float32, 길이 15600 (0.975초)
/// 출력 텐서 0: scores [N, 521]
class YamnetClassifier {
  static const int sampleRate = 16000;
  static const int inputLength = 15600;

  Interpreter? _interpreter;
  List<String> _labels = [];

  bool get isReady => _interpreter != null && _labels.isNotEmpty;

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/yamnet.tflite');

    final raw = await rootBundle.loadString('assets/yamnet_class_map.csv');
    final lines = raw.split('\n');
    _labels = [];
    for (int i = 1; i < lines.length; i++) {
      final line = lines[i].trim();
      if (line.isEmpty) continue;
      final parts = line.split(',');
      if (parts.length >= 3) {
        _labels.add(parts[2]);
      }
    }
  }

  /// PCM Float32 샘플(길이 15600)을 받아 분류 결과 반환
  List<MapEntry<String, double>> classify(Float32List samples) {
    final interp = _interpreter;
    if (interp == null) throw StateError('YAMNet interpreter not loaded');
    if (samples.length != inputLength) {
      throw ArgumentError('Input length must be $inputLength, got ${samples.length}');
    }

    final outputShape = interp.getOutputTensor(0).shape;
    final numFrames = outputShape[0];
    final numClasses = outputShape[1];

    final output = List.generate(numFrames, (_) => List<double>.filled(numClasses, 0.0));
    interp.run(samples, output);

    final avgScores = List<double>.filled(numClasses, 0.0);
    for (int f = 0; f < numFrames; f++) {
      for (int c = 0; c < numClasses; c++) {
        avgScores[c] += output[f][c];
      }
    }
    for (int c = 0; c < numClasses; c++) {
      avgScores[c] /= numFrames;
    }

    final entries = <MapEntry<String, double>>[];
    for (int c = 0; c < numClasses && c < _labels.length; c++) {
      entries.add(MapEntry(_labels[c], avgScores[c]));
    }
    entries.sort((a, b) => b.value.compareTo(a.value));
    return entries;
  }

  // ── M_chew: 씹는 소리 ─────────────────────────────
  double chewingScore(List<MapEntry<String, double>> all) {
    double best = 0.0;
    for (final e in all) {
      final label = e.key.toLowerCase();
      if (label.contains('chewing') ||
          label.contains('mastication') ||
          label.contains('biting') ||
          label.contains('crunch')) {
        if (e.value > best) best = e.value;
      }
    }
    return best;
  }

  // ── M_drink: 마시는 소리 ──────────────────────────
  //
  // 실기기 테스트(2026-05-15)로 확인된 실제 라벨 분포:
  //
  //   물 마시는 순간:
  //     Drip      10.9%  ← 물 입에 닿는 소리
  //     Glass     10.9%  ← 컵 소리  ★ 핵심 발견
  //     "Chink    8.2%   ← 컵 부딪힘 ★ 핵심 발견
  //     Liquid    5.9%
  //     Water     2.0%
  //     "Water tap 1.2%
  //     Pour      0.8%
  //     Fill(with liquid) 0.8%
  //
  //   마시는 행위 전후 배경:
  //     "Inside   8~15%  (항상 높게 나오는 배경 노이즈 → 단독 사용 금지)
  //     Speech    5~8%
  //
  // 전략:
  //   - Glass / Chink 를 핵심 라벨로 승격 (실측 근거)
  //   - Drip / Liquid / Water / Pour 도 포함
  //   - "Inside" 와 "Speech" 는 배경 노이즈이므로 제외
  //   - 핵심 라벨(glass/chink/gulp/slurp/swallow/gargling) 중
  //     하나라도 5% 이상이면 보너스 +0.10
  // ─────────────────────────────────────────────────
  double drinkingScore(List<MapEntry<String, double>> all) {
    double broad = 0.0;
    double core  = 0.0;

    for (final e in all) {
      final label = e.key.toLowerCase();

      // 핵심: 실측으로 확인된 마시는 행위 직접 라벨
      final isCore = label.contains('glass') ||    // ★ 실측 1위
          label.contains('chink') ||               // ★ 실측 확인
          label.contains('gulp') ||
          label.contains('slurp') ||
          label.contains('swallow') ||
          label.contains('gargling');

      // 넓음: 마시는 행위 간접 증거
      final isBroad = isCore ||
          label.contains('drip') ||               // 실측 1위 (broad)
          label.contains('liquid') ||             // 실측 확인
          label.contains('water') ||              // 실측 확인
          label.contains('pour') ||
          label.contains('fill') ||               // "Fill (with liquid)"
          label.contains('burping') ||
          label.contains('eructation') ||
          label.contains('throat') ||
          label.contains('wet') ||
          label.contains('digestive') ||
          label.contains('cough');
      // ※ 'inside', 'speech', 'breathing' 은 배경 노이즈이므로 제외

      if (isBroad && e.value > broad) broad = e.value;
      if (isCore  && e.value > core)  core  = e.value;
    }

    // 핵심 라벨이 5% 이상이면 보너스
    final bonus = core >= 0.05 ? 0.10 : 0.0;
    return (broad + bonus).clamp(0.0, 1.0);
  }

  /// 디버그용: 마시는 소리 관련 라벨 전체 반환
  List<MapEntry<String, double>> drinkDebugLabels(List<MapEntry<String, double>> all) {
    const keywords = [
      'gulp', 'slurp', 'swallow', 'gargling', 'glass', 'chink',
      'liquid', 'pour', 'water', 'drip', 'fill', 'burping',
      'eructation', 'throat', 'wet', 'digestive', 'cough',
      'breathing', 'speech', 'inside', 'silence', 'noise',
    ];
    return all.where((e) {
      final l = e.key.toLowerCase();
      return keywords.any((k) => l.contains(k));
    }).toList();
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
