import { useState, useCallback, useRef, useEffect } from "react";
import { useLanguage } from "@/contexts/LanguageContext";

/**
 * ChatGPT-style voice hook.
 *  - Click once → stays listening (auto-restarts after each utterance).
 *  - When browser fires `onend` (silence detected), transcript is sent via onSilence().
 *  - Mic pauses during TTS / AI thinking, resumes after.
 *  - Click again → stops entirely.
 */
export function useSpeechToText(onSilence?: (text: string) => void) {
  const { language } = useLanguage();
  const [isListening, setIsListening] = useState(false);
  const [interimText, setInterimText] = useState("");

  const recognitionRef = useRef<any>(null);
  const activeRef = useRef(false);         // true while voice mode is on
  const onSilenceRef = useRef(onSilence);  // always latest callback
  const finalRef = useRef("");             // accumulated final transcript

  // Keep the callback ref in sync
  useEffect(() => { onSilenceRef.current = onSilence; }, [onSilence]);

  const startSession = useCallback(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition || !activeRef.current) return;

    const recognition = new SpeechRecognition();
    recognition.lang = language.speechCode;
    recognition.interimResults = true;
    recognition.continuous = false; // browser handles silence → fires onend
    recognition.maxAlternatives = 1;
    recognitionRef.current = recognition;
    finalRef.current = "";

    recognition.onstart = () => {
      setIsListening(true);
      setInterimText("");
    };

    recognition.onresult = (event: any) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalRef.current = (finalRef.current + " " + t).trim();
        } else {
          interim = t;
        }
      }
      setInterimText(interim || finalRef.current);
    };

    recognition.onerror = (event: any) => {
      if (event.error !== "no-speech" && event.error !== "aborted") {
        console.error("Speech error:", event.error);
      }
    };

    recognition.onend = () => {
      setIsListening(false);
      setInterimText("");

      const text = finalRef.current.trim();
      finalRef.current = "";

      // Fire the callback with whatever was captured
      if (text && onSilenceRef.current && activeRef.current) {
        onSilenceRef.current(text);
        // Don't restart until the caller signals it's ready (via resumeListening)
        return;
      }

      // No text (noise / no-speech): immediately restart
      if (activeRef.current) {
        setTimeout(() => {
          if (activeRef.current) startSession();
        }, 100);
      }
    };

    try {
      recognition.start();
    } catch (e) {
      // Already started — ignore
    }
  }, [language.speechCode]);

  const startListening = useCallback(() => {
    activeRef.current = true;
    startSession();
  }, [startSession]);

  const stopListening = useCallback(() => {
    activeRef.current = false;
    recognitionRef.current?.abort();
    setIsListening(false);
    setInterimText("");
    finalRef.current = "";
  }, []);

  const pauseListening = useCallback(() => {
    recognitionRef.current?.abort();
    setIsListening(false);
  }, []);

  const resumeListening = useCallback(() => {
    if (activeRef.current) {
      setTimeout(() => {
        if (activeRef.current) startSession();
      }, 400);
    }
  }, [startSession]);

  useEffect(() => {
    return () => {
      activeRef.current = false;
      recognitionRef.current?.abort();
    };
  }, []);

  return { isListening, interimText, startListening, stopListening, pauseListening, resumeListening };
}

/**
 * Text-to-Speech — calls onEnd when finished.
 */
export function useTextToSpeech() {
  const { language } = useLanguage();
  const [isSpeaking, setIsSpeaking] = useState(false);

  const speak = useCallback(
    (text: string, overrideLang?: string, onEnd?: () => void) => {
      if (!window.speechSynthesis) { onEnd?.(); return; }
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = overrideLang || language.speechCode;
      utterance.rate = 0.95;
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => { setIsSpeaking(false); onEnd?.(); };
      utterance.onerror = () => { setIsSpeaking(false); onEnd?.(); };
      window.speechSynthesis.speak(utterance);
    },
    [language.speechCode]
  );

  const stop = useCallback(() => {
    window.speechSynthesis?.cancel();
    setIsSpeaking(false);
  }, []);

  return { isSpeaking, speak, stop };
}
