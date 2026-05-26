import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import 'audio_streamer.dart';
import 'yamnet_classifier.dart';
import 'drink_detector.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'YAMNet 식음 감지',
        theme: ThemeData(colorSchemeSeed: Colors.deepOrange, useMaterial3: true),
        home: const DetectorPage(),
      );
}

class DetectorPage extends StatefulWidget {
  const DetectorPage({super.key});
  @override
  State<DetectorPage> createState() => _DetectorPageState();
}

class _DetectorPageState extends State<DetectorPage> {
  static const double chewingThreshold = 0.30;

  final YamnetClassifier _classifier   = YamnetClassifier();
  final AudioStreamer    _streamer      = AudioStreamer();
  final DrinkDetector   _drinkDetector = DrinkDetector();

  bool _modelReady = false;
  bool _listening  = false;
  bool _debugMode  = true;

  // M_chew
  bool      _chewTriggered = false;
  double    _chewScore     = 0.0;
  DateTime? _chewLastTime;

  // M_drink
  bool      _drinkTriggered = false;
  DateTime? _drinkLastTime;

  // FFT 디버그
  double _bandRms      = 0.0;
  double _background   = 0.0;
  double _riseRatio    = 0.0;
  double _highRatio    = 0.0;
  double _midRatio     = 0.0;
  bool   _isCough      = false;
  bool   _inCooldown   = false;
  bool   _preBlocked   = false;
  int    _activeFrames = 0;

  List<MapEntry<String, double>> _topResults = [];

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    await _classifier.loadModel();
    await _streamer.init();
    _streamer.onChunk = _onChunk;
    setState(() => _modelReady = true);
  }

  void _onChunk(Float32List chunk) {
    if (!_listening) return;
    final now = DateTime.now();

    final all    = _classifier.classify(chunk);
    final cScore = _classifier.chewingScore(all);
    final event  = _drinkDetector.process(chunk);

    setState(() {
      _topResults  = all.take(5).toList();
      _chewScore   = cScore;
      _bandRms     = event.bandRms;
      _background  = event.background;
      _riseRatio   = event.riseRatio;
      _highRatio   = event.highRatio;
      _midRatio    = event.midRatio;
      _isCough     = event.isCough;
      _inCooldown  = event.inCooldown;
      _preBlocked  = event.preBlocked;
      _activeFrames = event.activeFrames;

      if (cScore >= chewingThreshold) {
        _chewTriggered = true;
        _chewLastTime  = now;
      }
      if (event.detected) {
        _drinkTriggered = true;
        _drinkLastTime  = now;
      }
    });
  }

  Future<void> _toggleListening() async {
    if (_listening) {
      await _streamer.stop();
      _drinkDetector.reset();
      setState(() => _listening = false);
      return;
    }
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('마이크 권한이 필요합니다')));
      return;
    }
    await _streamer.start();
    setState(() {
      _listening      = true;
      _chewTriggered  = false;
      _drinkTriggered = false;
      _topResults     = [];
    });
  }

  void _resetTriggers() => setState(() {
        _chewTriggered  = false;
        _drinkTriggered = false;
        _chewScore      = 0.0;
      });

  @override
  void dispose() {
    _streamer.stop();
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('YAMNet 식음 감지'),
        actions: [
          Row(children: [
            const Text('디버그', style: TextStyle(fontSize: 13)),
            Switch(value: _debugMode, onChanged: (v) => setState(() => _debugMode = v)),
          ]),
        ],
      ),
      body: !_modelReady
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // 상태 배너
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: _listening ? Colors.green.shade50 : Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: _listening ? Colors.green : Colors.grey),
                    ),
                    child: Text(
                      _listening ? '듣는 중... 음식을 씹거나 음료를 마셔보세요.' : '시작 버튼을 눌러 감지를 시작하세요.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: _listening ? Colors.green.shade800 : Colors.grey.shade600,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // M_chew 카드
                  _TriggerCard(
                    triggered: _chewTriggered,
                    label: 'M_chew  씹는 소리',
                    subLabel: 'YAMNet',
                    emoji: '🍽',
                    detectedText: '씹는 소리 감지됨!',
                    scoreText: '${(_chewScore * 100).toStringAsFixed(1)}%',
                    thresholdText: '임계값 30%',
                    aboveThreshold: _chewScore >= chewingThreshold,
                    lastTime: _chewLastTime,
                    activeColor: Colors.green,
                    extraText: null,
                  ),
                  const SizedBox(height: 12),

                  // M_drink 카드
                  _TriggerCard(
                    triggered: _drinkTriggered,
                    label: 'M_drink  마시는 소리',
                    subLabel: 'FFT 대역 필터 100~500Hz',
                    emoji: '🥤',
                    detectedText: '마시는 소리 감지됨!',
                    scoreText: '↑${_riseRatio.toStringAsFixed(1)}배',
                    thresholdText: '기준 3.0배',
                    aboveThreshold: _riseRatio >= 3.0,
                    lastTime: _drinkLastTime,
                    activeColor: Colors.blue,
                    extraText: _activeFrames > 0 ? '활성 $_activeFrames 프레임' : null,
                  ),
                  const SizedBox(height: 16),

                  // FFT 디버그 패널
                  if (_debugMode && _listening) ...[
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.blue.shade50,
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.blue.shade200),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('🔍 FFT 필터 실시간 지표',
                              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13)),
                          const SizedBox(height: 10),
                          _DebugGauge(
                            label: '대역 RMS (현재)',
                            value: _bandRms, maxValue: 0.005,
                            thresholdValue: 0.0005, color: Colors.blue,
                            valueText: _bandRms.toStringAsExponential(2),
                          ),
                          const SizedBox(height: 6),
                          _DebugGauge(
                            label: '대역 RMS (배경 추정)',
                            value: _background, maxValue: 0.005,
                            thresholdValue: null, color: Colors.grey,
                            valueText: _background.toStringAsExponential(2),
                          ),
                          const SizedBox(height: 6),
                          _DebugGauge(
                            label: '배경 대비 상승 배율  ← 핵심',
                            value: _riseRatio, maxValue: 10.0,
                            thresholdValue: 3.0, color: Colors.indigo,
                            valueText: '${_riseRatio.toStringAsFixed(2)}배',
                          ),
                          const SizedBox(height: 8),
                          // 프레임 카운터 표시
                          Row(children: [
                            const Text('활성 프레임: ', style: TextStyle(fontSize: 12)),
                            ...List.generate(3, (i) => Padding(
                              padding: const EdgeInsets.only(right: 4),
                              child: Container(
                                width: 20, height: 20,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: i < _activeFrames
                                      ? Colors.indigo
                                      : Colors.grey.shade300,
                                ),
                              ),
                            )),
                            const SizedBox(width: 8),
                            Text(
                              _activeFrames == 0
                                  ? '대기 중'
                                  : _activeFrames == 1
                                      ? '✓ 감지!'
                                      : _activeFrames >= 3
                                          ? '⚠ 노이즈 취소'
                                          : '활성 중',
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.bold,
                                color: _activeFrames == 1
                                    ? Colors.indigo
                                    : _activeFrames >= 3
                                        ? Colors.red
                                        : Colors.grey,
                              ),
                            ),
                          ]),
                          const SizedBox(height: 4),
                          Text(
                            '1프레임(≈1초) 활성 → 즉시 감지 | 3프레임 연속 활성 → 노이즈로 취소',
                            style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
                          ),
                          const SizedBox(height: 6),
                          _DebugGauge(
                            label: '중역/저역 비율  ← 말소리/키보드 차단 (0.50 초과)',
                            value: _midRatio, maxValue: 2.0,
                            thresholdValue: 0.50, color: Colors.orange,
                            valueText: _midRatio > 0.50
                                ? '${_midRatio.toStringAsFixed(2)} 차단'
                                : _midRatio.toStringAsFixed(2),
                          ),
                          const SizedBox(height: 6),
                          _DebugGauge(
                            label: '고역/저역 비율  ← 기침 차단 (0.45 이상)',
                            value: _highRatio, maxValue: 1.0,
                            thresholdValue: 0.45, color: Colors.red,
                            valueText: _isCough
                                ? '${_highRatio.toStringAsFixed(2)} ⚠ 기침'
                                : _inCooldown
                                    ? '${_highRatio.toStringAsFixed(2)} 🔕 쿨다운'
                                    : _highRatio.toStringAsFixed(2),
                          ),
                          const SizedBox(height: 6),
                          Container(
                            padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 8),
                            decoration: BoxDecoration(
                              color: _preBlocked
                                  ? Colors.deepOrange.withOpacity(0.15)
                                  : Colors.grey.shade100,
                              borderRadius: BorderRadius.circular(6),
                              border: Border.all(
                                color: _preBlocked ? Colors.deepOrange : Colors.grey.shade300),
                            ),
                            child: Row(children: [
                              Expanded(child: Text(
                                '선행 고에너지  ← 의자/충격음 차단 (직전 500ms 내 배율≥3.0)',
                                style: const TextStyle(fontSize: 12),
                              )),
                              Text(
                                _preBlocked ? '⚠ 차단 중' : '정상',
                                style: TextStyle(
                                  fontSize: 12, fontWeight: FontWeight.bold,
                                  color: _preBlocked ? Colors.deepOrange : Colors.grey),
                              ),
                            ]),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                  ],

                  // Top-5
                  if (_debugMode && _topResults.isNotEmpty) ...[
                    const Text('Top-5 YAMNet 결과',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    ..._topResults.map((e) => _ScoreRow(label: e.key, score: e.value)),
                    const SizedBox(height: 16),
                  ],

                  // 버튼
                  Row(children: [
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: _toggleListening,
                        icon: Icon(_listening ? Icons.stop : Icons.mic),
                        label: Text(_listening ? '중지' : '시작'),
                        style: FilledButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          backgroundColor: _listening ? Colors.red : Colors.deepOrange,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    ElevatedButton.icon(
                      onPressed: (_chewTriggered || _drinkTriggered) ? _resetTriggers : null,
                      icon: const Icon(Icons.refresh),
                      label: const Text('리셋'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
                      ),
                    ),
                  ]),
                ],
              ),
            ),
    );
  }
}

class _TriggerCard extends StatelessWidget {
  const _TriggerCard({
    required this.triggered, required this.label, required this.subLabel,
    required this.emoji, required this.detectedText, required this.scoreText,
    required this.thresholdText, required this.aboveThreshold,
    required this.lastTime, required this.activeColor, required this.extraText,
  });

  final bool triggered; final String label, subLabel, emoji;
  final String detectedText, scoreText, thresholdText;
  final bool aboveThreshold; final DateTime? lastTime;
  final Color activeColor; final String? extraText;

  String _fmt(DateTime t) =>
      '${t.hour.toString().padLeft(2,'0')}:${t.minute.toString().padLeft(2,'0')}:${t.second.toString().padLeft(2,'0')}';

  @override
  Widget build(BuildContext context) => AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: triggered ? activeColor.withOpacity(0.12) : Colors.grey.shade100,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
              color: triggered ? activeColor : Colors.grey.shade300,
              width: triggered ? 2 : 1),
        ),
        child: Row(children: [
          Text(emoji, style: const TextStyle(fontSize: 36)),
          const SizedBox(width: 12),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(label, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13)),
              Text(subLabel, style: TextStyle(fontSize: 11, color: Colors.grey.shade500)),
              const SizedBox(height: 4),
              Text(triggered ? detectedText : '미감지',
                  style: TextStyle(
                    color: triggered ? activeColor : Colors.grey,
                    fontWeight: triggered ? FontWeight.bold : FontWeight.normal)),
              if (lastTime != null)
                Text('마지막: ${_fmt(lastTime!)}',
                    style: const TextStyle(fontSize: 12, color: Colors.grey)),
              if (extraText != null)
                Text(extraText!, style: TextStyle(fontSize: 12, color: activeColor)),
            ]),
          ),
          Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
            Text(scoreText,
                style: TextStyle(
                  fontSize: 16, fontWeight: FontWeight.bold,
                  color: aboveThreshold ? activeColor : Colors.grey)),
            Text(thresholdText, style: const TextStyle(fontSize: 10, color: Colors.grey)),
          ]),
        ]),
      );
}

class _DebugGauge extends StatelessWidget {
  const _DebugGauge({
    required this.label, required this.value, required this.maxValue,
    required this.thresholdValue, required this.color, required this.valueText,
  });

  final String label; final double value, maxValue;
  final double? thresholdValue; final Color color; final String valueText;

  @override
  Widget build(BuildContext context) {
    final ratio = (value / maxValue).clamp(0.0, 1.0);
    final above = thresholdValue != null && value >= thresholdValue!;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        Expanded(child: Text(label, style: const TextStyle(fontSize: 12))),
        Text(valueText,
            style: TextStyle(
              fontSize: 12, fontWeight: FontWeight.bold,
              color: above ? color : Colors.grey.shade600)),
      ]),
      const SizedBox(height: 3),
      LayoutBuilder(builder: (ctx, constraints) {
        final w = constraints.maxWidth;
        return Stack(children: [
          Container(height: 8, decoration: BoxDecoration(
              color: Colors.grey.shade200, borderRadius: BorderRadius.circular(4))),
          Container(
            height: 8, width: ratio * w,
            decoration: BoxDecoration(
              color: above ? color : Colors.grey.shade400,
              borderRadius: BorderRadius.circular(4))),
          if (thresholdValue != null)
            Positioned(
              left: ((thresholdValue! / maxValue).clamp(0.0, 1.0) * w) - 1,
              child: Container(width: 2, height: 8, color: Colors.red.withOpacity(0.7))),
        ]);
      }),
    ]);
  }
}

class _ScoreRow extends StatelessWidget {
  const _ScoreRow({required this.label, required this.score});
  final String label; final double score;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(children: [
          Expanded(child: Text(label, style: const TextStyle(fontSize: 13))),
          SizedBox(
            width: 130,
            child: LinearProgressIndicator(
              value: score.clamp(0.0, 1.0),
              backgroundColor: Colors.grey.shade200,
              color: Colors.deepOrange)),
          const SizedBox(width: 8),
          Text('${(score * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
        ]),
      );
}
