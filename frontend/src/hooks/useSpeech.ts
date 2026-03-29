import { useState, useCallback, useRef, useEffect } from "react";
import { useLanguage } from "@/contexts/LanguageContext";

const SPEECH_LANG_FALLBACKS: Record<string, string[]> = {
  "hi-IN": ["hi-IN", "hi", "en-IN", "en-US"],
  "kok-IN": ["kok-IN", "hi-IN", "mr-IN", "en-IN"],
  "kn-IN": ["kn-IN", "te-IN", "hi-IN", "en-IN"],
  "doi-IN": ["doi-IN", "hi-IN", "en-IN"],
  "brx-IN": ["brx-IN", "as-IN", "bn-IN", "hi-IN", "en-IN"],
  "ur-IN": ["ur-IN", "ur", "hi-IN", "en-IN"],
  "ta-IN": ["ta-IN", "ta", "hi-IN", "en-IN"],
  "ks-IN": ["ks-IN", "ur-IN", "hi-IN", "en-IN"],
  "as-IN": ["as-IN", "bn-IN", "hi-IN", "en-IN"],
  "bn-IN": ["bn-IN", "bn", "hi-IN", "en-IN"],
  "mr-IN": ["mr-IN", "hi-IN", "en-IN"],
  "sd-IN": ["sd-IN", "ur-IN", "hi-IN", "en-IN"],
  "mai-IN": ["mai-IN", "hi-IN", "en-IN"],
  "pa-IN": ["pa-IN", "pa", "hi-IN", "en-IN"],
  "ml-IN": ["ml-IN", "ta-IN", "hi-IN", "en-IN"],
  "mni-IN": ["mni-IN", "bn-IN", "hi-IN", "en-IN"],
  "te-IN": ["te-IN", "kn-IN", "hi-IN", "en-IN"],
  "sa-IN": ["sa-IN", "hi-IN", "en-IN"],
  "ne-NP": ["ne-NP", "hi-IN", "en-IN"],
  "sat-IN": ["sat-IN", "hi-IN", "en-IN"],
  "gu-IN": ["gu-IN", "hi-IN", "en-IN"],
  "or-IN": ["or-IN", "bn-IN", "hi-IN", "en-IN"],
  "en-IN": ["en-IN", "en-GB", "en-US", "hi-IN"],
};

const getSpeechCandidates = (requested: string): string[] => {
  return SPEECH_LANG_FALLBACKS[requested] || [requested, "hi-IN", "en-IN", "en-US"];
};

const pickVoiceForLanguage = (requested: string): SpeechSynthesisVoice | null => {
  const voices = window.speechSynthesis?.getVoices?.() || [];
  if (!voices.length) {
    return null;
  }

  const candidates = getSpeechCandidates(requested);
  const matched = candidates
    .map((candidate) => {
      const exact = voices.find((v) => v.lang.toLowerCase() === candidate.toLowerCase());
      if (exact) return exact;

      const base = candidate.split("-")[0].toLowerCase();
      return voices.find((v) => v.lang.toLowerCase().startsWith(`${base}-`));
    })
    .find(Boolean);

  return matched || voices[0] || null;
};

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
  const recognitionLangRef = useRef(language.speechCode);

  useEffect(() => {
    recognitionLangRef.current = getSpeechCandidates(language.speechCode)[0];
  }, [language.speechCode]);

  // Keep the callback ref in sync
  useEffect(() => { onSilenceRef.current = onSilence; }, [onSilence]);

  // If user changes language while mic is active, restart recognizer with new language.
  useEffect(() => {
    if (!activeRef.current) return;
    recognitionRef.current?.abort();
    setTimeout(() => {
      if (activeRef.current) startSession();
    }, 120);
  }, [language.speechCode]);

  const startSession = useCallback(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition || !activeRef.current) return;

    const recognition = new SpeechRecognition();
    recognition.lang = recognitionLangRef.current;
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
      if (event.error === "language-not-supported") {
        const candidates = getSpeechCandidates(language.speechCode);
        const currentIndex = candidates.findIndex((code) => code === recognitionLangRef.current);
        const next = candidates[currentIndex + 1];
        if (next) {
          recognitionLangRef.current = next;
        }
      }

      // Hard-stop voice mode for permission/audio errors to avoid infinite "Listening..." state.
      if (event.error === "not-allowed" || event.error === "service-not-allowed" || event.error === "audio-capture") {
        activeRef.current = false;
        setIsListening(false);
        setInterimText("");
        return;
      }

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

  useEffect(() => {
    // On some browsers voices load asynchronously; this primes the voice list.
    window.speechSynthesis?.getVoices?.();
  }, []);

  const speak = useCallback(
    (text: string, overrideLang?: string, onEnd?: () => void) => {
      if (!window.speechSynthesis) { onEnd?.(); return; }
      if (!text || !text.trim()) { onEnd?.(); return; }
      window.speechSynthesis.cancel();
      window.speechSynthesis.resume();
      const utterance = new SpeechSynthesisUtterance(text);
      const requestedLang = overrideLang || language.speechCode;
      const fallbackLang = getSpeechCandidates(requestedLang)[0];
      utterance.lang = fallbackLang;
      const voice = pickVoiceForLanguage(requestedLang);
      if (voice) {
        utterance.voice = voice;
        utterance.lang = voice.lang;
      }
      utterance.rate = 0.95;
      utterance.volume = 1;
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => { setIsSpeaking(false); onEnd?.(); };
      utterance.onerror = () => {
        // Retry once with a safe English fallback voice.
        try {
          const retry = new SpeechSynthesisUtterance(text);
          retry.lang = "en-IN";
          retry.rate = 0.95;
          retry.volume = 1;
          retry.onend = () => { setIsSpeaking(false); onEnd?.(); };
          retry.onerror = () => { setIsSpeaking(false); onEnd?.(); };
          window.speechSynthesis.speak(retry);
        } catch {
          setIsSpeaking(false);
          onEnd?.();
        }
      };
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
