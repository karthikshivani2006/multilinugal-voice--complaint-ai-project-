import { LANGUAGES, useLanguage } from "@/contexts/LanguageContext";
import { Link } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import { FileText, Search, Mic, Phone, AlertTriangle, Users, Globe } from "lucide-react";
import MicButton from "@/components/MicButton";
import { useSpeechToText, useTextToSpeech } from "@/hooks/useSpeech";
import { useCallback, useEffect, useRef, useState } from "react";
import BrandLogo from "@/components/BrandLogo";
import { sendChatMessage, resetChatSession, type ChatMessage } from "@/services/api";

export default function HomePage() {
  const { t, isLanguageSelected, language, setLanguage } = useLanguage();
  const navigate = useNavigate();
  const [voiceActive, setVoiceActive] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);
  const [lastTranscript, setLastTranscript] = useState("");
  const [assistantReply, setAssistantReply] = useState("");
  const { speak, stop: stopSpeaking } = useTextToSpeech();
  const greetedRef = useRef(false);
  const [showTranscript, setShowTranscript] = useState(false);
  const historyRef = useRef<ChatMessage[]>([]);
  const voiceActiveRef = useRef(false);
  const resumeListeningRef = useRef<(() => void) | null>(null);

  // Ensure Home assistant starts a new backend session on each page load/refresh.
  useEffect(() => {
    resetChatSession();
    historyRef.current = [];
    setLastTranscript("");
    setAssistantReply("");
    setShowTranscript(false);
  }, []);

  const findLanguageFromSpeech = useCallback((rawText: string) => {
    const normalized = rawText.trim().toLowerCase();
    if (!normalized) return null;

    const compact = normalized.replace(/\s+/g, "");
    const languageAliases: Record<string, string[]> = {
      en: ["english", "inglish", "en"],
      hi: ["hindi", "hindhi", "hin", "हिंदी", "हिन्दी"],
      te: ["telugu", "తెలుగు"],
      ta: ["tamil", "தமிழ்"],
      kn: ["kannada", "ಕನ್ನಡ"],
      ml: ["malayalam", "മലയാളം"],
      mr: ["marathi", "मराठी"],
      bn: ["bengali", "bangla", "বাংলা"],
      gu: ["gujarati", "ગુજરાતી"],
      pa: ["punjabi", "ਪੰਜਾਬੀ"],
      ur: ["urdu", "اردو"],
      or: ["odia", "oriya", "ଓଡ଼ିଆ"],
      as: ["assamese", "অসমীয়া"],
      ne: ["nepali", "नेपाली"],
      kok: ["konkani", "कोंकणी"],
      ks: ["kashmiri", "कॉशुर"],
      sa: ["sanskrit", "संस्कृतम्"],
      sd: ["sindhi", "سنڌي"],
      mai: ["maithili", "मैथिली"],
      doi: ["dogri", "डोगरी"],
      brx: ["bodo", "बड़ो"],
      mni: ["manipuri", "meitei", "মৈতৈ"],
      sat: ["santali", "ᱥᱟᱱᱛᱟᱲᱤ"],
    };

    for (const lang of LANGUAGES) {
      const aliases = languageAliases[lang.code] || [];
      const direct = [lang.name.toLowerCase(), lang.nativeName.toLowerCase(), ...aliases];
      if (direct.some((item) => normalized.includes(item.toLowerCase()))) {
        return lang;
      }
      if (direct.some((item) => compact.includes(item.toLowerCase().replace(/\s+/g, "")))) {
        return lang;
      }
    }

    return null;
  }, []);

  const maybeAutoSwitchLanguage = useCallback((text: string) => {
    const detected = findLanguageFromSpeech(text);
    if (!detected) return language;
    if (language.code === "en" && detected.code !== "en") {
      setLanguage(detected);
      return detected;
    }
    return language;
  }, [findLanguageFromSpeech, language, setLanguage]);

  const localizeHomeAssistantReply = useCallback((raw: string, userText: string) => {
    const text = (raw || "").trim();
    if (!text) return t("sorryError");

    const isGenericBackendFallback = /please continue and share your complaint details clearly\.?/i.test(text)
      || /ai service is temporarily busy/i.test(text)
      || /please share your full name|please share your 10-digit phone number|please share your email address|please share your complete address/i.test(text);

    if (!isGenericBackendFallback) {
      return text;
    }

    const wantsHelp = /(don't know|dont know|not sure|no idea|help|guide|explain|naaku teliyadu|naku teliyadu|teliyadu)/i.test(userText);
    if (wantsHelp) {
      if (language.code === "te") {
        return "నేను మీకు దశలవారీగా సహాయం చేస్తాను. ఫిర్యాదు నమోదు చేయడానికి మీ పూర్తి పేరు, ఫోన్ నంబర్, ఇమెయిల్, చిరునామా, సంఘటన వివరాలు చెప్పండి.";
      }
      return `${t("greeting")} ${t("fileComplaint")}.`;
    }

    return t("greeting");
  }, [language.code, t]);

  const handleVoiceMessage = useCallback(async (text: string) => {
    const cleaned = text.trim();
    if (!cleaned) {
      return;
    }

    const effectiveLanguage = maybeAutoSwitchLanguage(cleaned);

    setLastTranscript(cleaned);
    setShowTranscript(true);

    if (!isLanguageSelected) {
      const detected = findLanguageFromSpeech(cleaned);
      if (!detected) {
        const prompt = t("languagePrompt");
        setAssistantReply(prompt);
        speak(prompt, "en-IN", () => {
          if (voiceActiveRef.current) {
            resumeListeningRef.current?.();
          }
        });
        return;
      }

      setLanguage(detected);
      historyRef.current = [];
      const selectedReply = `${detected.nativeName} ${t("languageSelected")}`;
      setAssistantReply(selectedReply);
      speak(selectedReply, detected.speechCode, () => {
        if (voiceActiveRef.current) {
          resumeListeningRef.current?.();
        }
      });
      return;
    }

    if (/\b(hi|hello|hey|hii|helo|good morning|good evening|namaste|నమస్తే|నమస్కారం)\b/i.test(cleaned) && historyRef.current.length === 0) {
      const greetReply = language.code === "en" ? "Hello! What can I do for you today?" : t("greeting");
      setAssistantReply(greetReply);
      speak(greetReply, effectiveLanguage.speechCode, () => {
        if (voiceActiveRef.current) {
          resumeListeningRef.current?.();
        }
      });
      return;
    }

    // If user asks to file a complaint, move them directly to complaint workflow.
    if (/(file|register|submit|raise)\s+.*(complaint|report)|\bcomplaint\b/i.test(cleaned)) {
      speak(t("fileComplaint"), effectiveLanguage.speechCode, () => {
        navigate("/complaint");
      });
      return;
    }

    const userMsg: ChatMessage = { role: "user", content: cleaned };
    const nextMessages = [...historyRef.current, userMsg];
    historyRef.current = nextMessages;

    setIsProcessingVoice(true);
    try {
      const apiRes = await sendChatMessage(nextMessages, effectiveLanguage.name);
      const reply = localizeHomeAssistantReply(apiRes.response || t("sorryError"), cleaned);
      const assistantMsg: ChatMessage = { role: "assistant", content: reply };
      historyRef.current = [...nextMessages, assistantMsg];
      setAssistantReply(reply);
      speak(reply, effectiveLanguage.speechCode, () => {
        if (voiceActiveRef.current) {
          resumeListeningRef.current?.();
        }
      });
    } catch {
      const fallback = t("sorryError");
      setAssistantReply(fallback);
      speak(fallback, effectiveLanguage.speechCode, () => {
        if (voiceActiveRef.current) {
          resumeListeningRef.current?.();
        }
      });
    } finally {
      setIsProcessingVoice(false);
    }
  }, [findLanguageFromSpeech, isLanguageSelected, navigate, setLanguage, speak, t, maybeAutoSwitchLanguage, language.code]);

  const {
    isListening,
    startListening,
    stopListening,
    pauseListening,
    resumeListening,
  } = useSpeechToText(handleVoiceMessage);

  useEffect(() => {
    resumeListeningRef.current = resumeListening;
  }, [resumeListening]);

  useEffect(() => {
    voiceActiveRef.current = voiceActive;
  }, [voiceActive]);

  // Greet user when language is selected
  useEffect(() => {
    if (isLanguageSelected && !greetedRef.current) {
      greetedRef.current = true;
      setTimeout(() => speak(t("greeting"), language.speechCode), 500);
    }
  }, [isLanguageSelected, language.speechCode, speak, t]);

  const toggleMic = () => {
    if (voiceActive) {
      setVoiceActive(false);
      stopListening();
      stopSpeaking();
    } else {
      setVoiceActive(true);
      setShowTranscript(false);
      setAssistantReply("");

      if (!isLanguageSelected) {
        const prompt = t("languagePrompt");
        setAssistantReply(prompt);
        setShowTranscript(true);
        // Avoid capturing the app's own prompt audio as user input.
        pauseListening();
        speak(prompt, language.speechCode, () => {
          if (voiceActiveRef.current) resumeListeningRef.current?.();
        });
      } else {
        startListening();
      }
    }
  };

  const stats = [
    { icon: FileText, value: "2.5L+", label: t("fileComplaint") },
    { icon: Users, value: "1.8L+", label: t("resolved") },
    { icon: Globe, value: "23", label: t("language") },
    { icon: Phone, value: "1930", label: t("helpline") },
  ];

  return (
    <div className="min-h-[calc(100vh-120px)]">
      {/* Hero Section */}
      <section className="gov-gradient text-primary-foreground py-16 md:py-24 relative overflow-hidden">
        {/* Subtle pattern overlay */}
        <div className="absolute inset-0 opacity-5" style={{
          backgroundImage: "radial-gradient(circle at 2px 2px, white 1px, transparent 0)",
          backgroundSize: "32px 32px",
        }} />

        <div className="max-w-[1400px] mx-auto px-4 relative z-10">
          <div className="max-w-3xl mx-auto text-center">
            <div className="flex justify-center mb-6 animate-reveal-up">
              <BrandLogo sizeClassName="w-24 h-24 md:w-28 md:h-28" className="shadow-xl" />
            </div>
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-primary-foreground/20 bg-primary-foreground/5 mb-6 animate-reveal-up">
              <AlertTriangle className="w-4 h-4 text-accent" />
              <span className="text-sm text-primary-foreground/80">{t("helpline")}</span>
            </div>

            <h1 className="text-3xl md:text-5xl font-bold leading-[1.1] tracking-tight mb-4 text-balance animate-reveal-up" style={{ animationDelay: "80ms" }}>
              {t("cyberGuard")}
            </h1>
            <p className="text-lg md:text-xl text-primary-foreground/70 mb-8 text-pretty animate-reveal-up" style={{ animationDelay: "160ms" }}>
              {t("tagline")} — {t("govIndia")}
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-reveal-up" style={{ animationDelay: "240ms" }}>
              <Link
                to="/complaint"
                className="inline-flex items-center gap-2 px-8 py-3.5 rounded-lg bg-accent text-accent-foreground font-semibold text-base shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
              >
                <FileText className="w-5 h-5" />
                {t("fileComplaint")}
              </Link>
              <Link
                to="/track"
                className="inline-flex items-center gap-2 px-8 py-3.5 rounded-lg bg-primary-foreground/10 text-primary-foreground font-semibold text-base border border-primary-foreground/20 hover:bg-primary-foreground/15 transition-all duration-200 active:scale-[0.98]"
              >
                <Search className="w-5 h-5" />
                {t("trackComplaint")}
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="bg-card border-b border-border shadow-sm">
        <div className="max-w-[1400px] mx-auto px-4 py-8 grid grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((stat, i) => (
            <div key={stat.label} className="text-center animate-reveal-up" style={{ animationDelay: `${i * 100}ms` }}>
              <stat.icon className="w-8 h-8 mx-auto mb-2 text-accent" />
              <div className="text-2xl font-bold text-foreground tabular-nums">{stat.value}</div>
              <div className="text-sm text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="max-w-[1400px] mx-auto px-4 py-16">
        <h2 className="text-2xl font-bold text-center mb-12 text-balance">{t("cyberGuard")}</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {[
            { step: "01", title: t("selectLanguage"), desc: t("languagePrompt"), icon: Globe },
            { step: "02", title: t("description"), desc: t("aiAssistant"), icon: Mic },
            { step: "03", title: t("trackComplaint"), desc: t("trackStatus"), icon: Search },
          ].map((item, i) => (
            <div
              key={item.step}
              className="bg-card rounded-xl p-8 shadow-sm border border-border hover:shadow-md transition-shadow duration-300 animate-reveal-up"
              style={{ animationDelay: `${i * 120}ms` }}
            >
              <div className="text-xs font-bold text-accent mb-3 tracking-wider">STEP {item.step}</div>
              <item.icon className="w-10 h-10 text-primary mb-4" />
              <h3 className="text-lg font-semibold mb-2">{item.title}</h3>
              <p className="text-sm text-muted-foreground text-pretty">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Voice Transcript Overlay */}
      {showTranscript && lastTranscript && (
        <div className="fixed bottom-28 right-8 max-w-sm bg-card rounded-xl shadow-2xl border border-border p-4 z-50 animate-reveal-up">
          <p className="text-sm text-muted-foreground mb-1">{t("description")}</p>
          <p className="text-foreground font-medium">{lastTranscript}</p>
          {(assistantReply || isProcessingVoice) && (
            <>
              <p className="text-sm text-muted-foreground mt-3 mb-1">{t("aiAssistant")}</p>
              <p className="text-foreground font-medium">
                {isProcessingVoice ? "..." : assistantReply}
              </p>
            </>
          )}
          <button
            onClick={() => setShowTranscript(false)}
            className="text-xs text-accent mt-2 hover:underline"
          >
            X
          </button>
        </div>
      )}

      {/* Floating Mic Button */}
      <div className="fixed bottom-8 right-8 z-50">
        <MicButton isListening={voiceActive} onToggle={toggleMic} size="lg" />
        {voiceActive && (
          <div className="absolute -top-8 left-1/2 -translate-x-1/2 whitespace-nowrap text-xs bg-foreground text-background px-3 py-1 rounded-full">
            {isListening ? t("listening") : isProcessingVoice ? t("thinking") : isLanguageSelected ? t("voiceOn") : t("waitingForLanguage")}
          </div>
        )}
      </div>
    </div>
  );
}

function getTranslationLabel(_langCode: string, translated: string, fallback: string) {
  return translated?.trim() ? translated : fallback;
}
