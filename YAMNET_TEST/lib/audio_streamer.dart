import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_sound/flutter_sound.dart';

/// 마이크에서 16kHz mono PCM Int16 스트림을 받아
/// YAMNet 입력 길이(15600 샘플)만큼 모아서 콜백으로 넘겨주는 클래스.
///
/// flutter_sound 9.28+ 에서는 toStream 파라미터가
/// StreamSink<Uint8List> 형태로 바뀌었습니다.
class AudioStreamer {
  static const int sampleRate = 16000;
  static const int chunkSize = 15600; // YAMNet 0.975초

  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  StreamController<Uint8List>? _pcmController;
  StreamSubscription<Uint8List>? _sub;

  // 누적 버퍼 (Float32로 변환된 샘플)
  final List<double> _buffer = [];

  bool _isRunning = false;
  bool get isRunning => _isRunning;

  /// 청크가 모일 때마다 호출되는 콜백
  void Function(Float32List chunk)? onChunk;

  Future<void> init() async {
    await _recorder.openRecorder();
  }

  Future<void> start() async {
    if (_isRunning) return;
    _buffer.clear();

    _pcmController = StreamController<Uint8List>();

    // PCM 16-bit 스트림 시작 (Uint8List 형태로 들어옴)
    await _recorder.startRecorder(
      codec: Codec.pcm16,
      numChannels: 1,
      sampleRate: sampleRate,
      toStream: _pcmController!.sink,
    );

    _sub = _pcmController!.stream.listen(_onPcmData);

    _isRunning = true;
  }

  void _onPcmData(Uint8List pcm16Bytes) {
    // Int16 little-endian -> Float32 [-1, 1]
    final byteData = ByteData.sublistView(pcm16Bytes);
    final samplesCount = pcm16Bytes.length ~/ 2;

    for (int i = 0; i < samplesCount; i++) {
      final s = byteData.getInt16(i * 2, Endian.little);
      _buffer.add(s / 32768.0);
    }

    // 15600 샘플이 모일 때마다 분류 콜백 호출
    while (_buffer.length >= chunkSize) {
      final chunk = Float32List.fromList(_buffer.sublist(0, chunkSize));
      _buffer.removeRange(0, chunkSize);
      onChunk?.call(chunk);
    }
  }

  Future<void> stop() async {
    if (!_isRunning) return;
    await _recorder.stopRecorder();
    await _sub?.cancel();
    await _pcmController?.close();
    _sub = null;
    _pcmController = null;
    _buffer.clear();
    _isRunning = false;
  }

  Future<void> dispose() async {
    await stop();
    await _recorder.closeRecorder();
  }
}
