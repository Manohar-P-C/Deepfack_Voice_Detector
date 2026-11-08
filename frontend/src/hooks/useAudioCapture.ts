/**
 * Custom hook for capturing microphone audio via Web Audio API
 * or simulating streaming from a file / synthesizer.
 * Converts to 16kHz mono Float32 PCM chunks and feeds them to the callback.
 */

import { useCallback, useRef, useState } from 'react';

interface UseAudioCaptureProps {
  onAudioChunk: (pcmData: ArrayBuffer) => void;
  sampleRate?: number;
}

export function useAudioCapture({ onAudioChunk, sampleRate = 16000 }: UseAudioCaptureProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | AudioWorkletNode | null>(null);
  const demoIntervalRef = useRef<number | null>(null);

  // Resample buffer to target sample rate (16kHz)
  const resampleTo16k = (audioBuffer: Float32Array, inputSampleRate: number): Float32Array => {
    if (inputSampleRate === sampleRate) return audioBuffer;
    const ratio = inputSampleRate / sampleRate;
    const newLength = Math.round(audioBuffer.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const srcIndex = Math.min(Math.round(i * ratio), audioBuffer.length - 1);
      result[i] = audioBuffer[srcIndex];
    }
    return result;
  };

  const startMicCapture = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: true,
        },
      });

      mediaStreamRef.current = stream;
      const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      // Using 4096 buffer size (~92ms at 44.1kHz or ~256ms at 16kHz)
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);

        // Calculate RMS for visual audio level
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        setAudioLevel(Math.min(1, rms * 5));

        // Resample to 16kHz Float32
        const resampled = resampleTo16k(inputData, audioCtx.sampleRate);
        onAudioChunk(resampled.buffer as ArrayBuffer);
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
      setIsRecording(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Microphone access denied or unavailable';
      setError(msg);
      setIsRecording(false);
    }
  }, [onAudioChunk, sampleRate]);

  const stopMicCapture = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setIsRecording(false);
    setAudioLevel(0);
  }, []);

  // Demo file audio streamer: decodes an uploaded or sample audio file and streams in real-time
  const startFileSimulation = useCallback(async (file: File | Blob) => {
    try {
      setError(null);
      stopMicCapture();

      const arrayBuffer = await file.arrayBuffer();
      const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      audioCtx.close();

      const channelData = decodedBuffer.getChannelData(0);
      const resampled = resampleTo16k(channelData, decodedBuffer.sampleRate);

      setIsRecording(true);

      // Stream in 200ms chunks (3200 samples at 16kHz)
      const chunkSize = 3200;
      let offset = 0;

      if (demoIntervalRef.current) clearInterval(demoIntervalRef.current);

      demoIntervalRef.current = window.setInterval(() => {
        if (offset >= resampled.length) {
          offset = 0; // Loop or finish
        }
        const slice = resampled.subarray(offset, Math.min(offset + chunkSize, resampled.length));
        offset += chunkSize;

        // Level
        let sum = 0;
        for (let i = 0; i < slice.length; i++) {
          sum += slice[i] * slice[i];
        }
        setAudioLevel(Math.min(1, Math.sqrt(sum / slice.length) * 4));

        onAudioChunk(slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength) as ArrayBuffer);
      }, 200);

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to decode audio file';
      setError(msg);
    }
  }, [onAudioChunk, stopMicCapture]);

  // Synthetic speech wave generator for demo mode when no file is uploaded
  const startSyntheticStreaming = useCallback(() => {
    stopMicCapture();
    setIsRecording(true);

    const chunkSize = 3200; // 200ms at 16kHz
    let t = 0;

    demoIntervalRef.current = window.setInterval(() => {
      const pcm = new Float32Array(chunkSize);
      // Generate synthetic human vowel harmonics + formant frequencies
      for (let i = 0; i < chunkSize; i++) {
        const time = t + i / 16000;
        const f0 = 130 + 15 * Math.sin(2 * Math.PI * 1.5 * time); // pitch variation
        const f1 = 600;
        const f2 = 1400;
        const harmonic = 0.3 * Math.sin(2 * Math.PI * f0 * time)
          + 0.15 * Math.sin(2 * Math.PI * f1 * time)
          + 0.08 * Math.sin(2 * Math.PI * f2 * time)
          + (Math.random() - 0.5) * 0.02; // breath noise
        pcm[i] = harmonic;
      }
      t += chunkSize / 16000;
      setAudioLevel(0.4 + Math.sin(t * 3) * 0.2);
      onAudioChunk(pcm.buffer);
    }, 200);
  }, [onAudioChunk, stopMicCapture]);

  const stopAll = useCallback(() => {
    stopMicCapture();
    if (demoIntervalRef.current) {
      clearInterval(demoIntervalRef.current);
      demoIntervalRef.current = null;
    }
    setIsRecording(false);
    setAudioLevel(0);
  }, [stopMicCapture]);

  return {
    isRecording,
    audioLevel,
    error,
    startMicCapture,
    stopMicCapture,
    startFileSimulation,
    startSyntheticStreaming,
    stopAll,
  };
}
