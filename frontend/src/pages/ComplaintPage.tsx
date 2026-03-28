import { useState, useRef, useEffect, useCallback } from "react";
import { useLanguage, LANGUAGES } from "@/contexts/LanguageContext";
import { useSpeechToText, useTextToSpeech } from "@/hooks/useSpeech";
import { sendChatMessage, submitComplaint, type ChatMessage, type ComplaintFormData } from "@/services/api";
import { COMPLAINT_TYPES, PLATFORMS } from "@/data/complaintTypes";
import MicButton from "@/components/MicButton";
import { Send, Upload, CheckCircle, Volume2 } from "lucide-react";

export default function ComplaintPage() {
  const { t, language, setLanguage, isLanguageSelected } = useLanguage();

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  // Always read the latest messages from a ref to avoid stale closures
  const messagesRef = useRef<ChatMessage[]>([]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  // Form state
  const [form, setForm] = useState<ComplaintFormData>({
    fullName: "", phone: "", email: "", address: "",
    incidentType: "", dateTime: "", description: "",
    amountLost: "", transactionId: "", suspectDetails: "",
    suspectVpa: "", suspectPhone: "", suspectBankAccount: "",
    platform: "", language: language.code,
  });
  const [evidenceFiles, setEvidenceFiles] = useState<File[]>([]);
  const [idFiles, setIdFiles] = useState<File[]>([]);
  const [submitted, setSubmitted] = useState(false);
  const [ticketId, setTicketId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");

  // Voice mode state
  const [voiceActive, setVoiceActive] = useState(false); // is mic toggled ON?

  const isFormValid = Boolean(
    form.fullName && form.phone && form.email && form.address &&
    form.incidentType && form.dateTime && form.description && form.platform
  );

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ─── Core send + voice loop ─────────────────────────────────────────
  // Forward declared as refs so the voice callbacks can always reference the latest
  const sendMessageRef = useRef<((text: string) => Promise<void>) | null>(null);

  const { speak, isSpeaking, stop: stopSpeaking } = useTextToSpeech();

  // onSilence is called by useSpeechToText when the user pauses speaking
  const onSilence = useCallback((text: string) => {
    sendMessageRef.current?.(text);
  }, []);

  const {
    isListening,
    interimText,
    startListening,
    stopListening,
    pauseListening,
    resumeListening,
  } = useSpeechToText(voiceActive ? onSilence : undefined);


  // Keep refs so sendMessage can always read latest values without stale closures
  const voiceActiveRef = useRef(false);
  useEffect(() => { voiceActiveRef.current = voiceActive; }, [voiceActive]);
  const resumeListeningRef = useRef(resumeListening);
  useEffect(() => { resumeListeningRef.current = resumeListening; }, [resumeListening]);
  // Note: mic pausing/resuming is driven entirely by the speak() onEnd callback
  // and onSilence — no useEffect needed here (they caused chatLoading to get stuck).

  const applyCollectedFields = useCallback((fields?: Record<string, string>) => {
    if (!fields || Object.keys(fields).length === 0) return;
    setForm((prev) => ({
      ...prev,
      fullName: fields.full_name || prev.fullName,
      phone: fields.phone_number || prev.phone,
      email: fields.email || prev.email,
      address: fields.address || prev.address,
      incidentType: fields.complaint_type || prev.incidentType,
      dateTime: fields.date_time || prev.dateTime,
      description: fields.description || prev.description,
      amountLost: fields.amount_lost || prev.amountLost,
      transactionId: fields.transaction_id || prev.transactionId,
      suspectDetails: fields.suspect_details || prev.suspectDetails,
      platform: fields.platform || prev.platform,
      suspectVpa: fields.suspect_vpa || prev.suspectVpa,
      suspectPhone: fields.suspect_phone || prev.suspectPhone,
      suspectBankAccount: fields.suspect_bank_account || prev.suspectBankAccount,
    }));
  }, []);

  const handleSubmit = useCallback(async () => {
    setSubmitError("");
    setSubmitting(true);
    try {
      const res = await submitComplaint({ ...form, evidenceFiles, idProofFiles: idFiles, language: language.code });
      setTicketId(res.ticketId);
      setSubmitted(true);
      speak(`${t("yourComplaintFiled")} ${res.ticketId}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : t("failedSubmit");
      setSubmitError(message);
    }
    setSubmitting(false);
  }, [form, evidenceFiles, idFiles, language.code, speak, t]);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || chatLoading) return;

    const currentMessages = messagesRef.current;
    const userMsg: ChatMessage = { role: "user", content: text.trim() };
    const newMessages = [...currentMessages, userMsg];
    setMessages(newMessages);
    messagesRef.current = newMessages;
    setInputText("");

    const resumeIfVoice = () => {
      if (voiceActiveRef.current) resumeListeningRef.current?.();
    };

    // Language selection
    if (!isLanguageSelected) {
      const matchedLang = LANGUAGES.find(
        (l: any) => text.toLowerCase().includes(l.name.toLowerCase()) || text.includes(l.nativeName)
      );
      if (matchedLang) {
        setLanguage(matchedLang);
        setChatLoading(true);
        try {
          const apiRes = await sendChatMessage(newMessages, matchedLang.name);
          const assistantMsg: ChatMessage = { role: "assistant", content: apiRes.response };
          setMessages((prev) => [...prev, assistantMsg]);
          messagesRef.current = [...newMessages, assistantMsg];
          applyCollectedFields(apiRes.collected_fields);
          speak(apiRes.response, matchedLang.speechCode, resumeIfVoice);
        } catch {
          setMessages((prev) => [...prev, { role: "assistant", content: "Sorry, an error occurred." }]);
          resumeIfVoice();
        } finally {
          setChatLoading(false);
        }
        return;
      } else {
        const reply = "Please choose your language. For example: English, Hindi, or Tamil.";
        setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
        speak(reply, undefined, resumeIfVoice);
        return;
      }
    }

    // Submit trigger via voice
    const submitRegex = /^(submit|file|confirm|register|submit the form|file my complaint)$/i;
    if (submitRegex.test(text.trim()) && isFormValid) {
      const reply = "Understood. Submitting your complaint now.";
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
      speak(reply, undefined, resumeIfVoice);
      handleSubmit();
      return;
    } else if (submitRegex.test(text.trim()) && !isFormValid) {
      const reply = "Some required fields are still missing. Please describe your full name, what happened, and the platform.";
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
      speak(reply, undefined, resumeIfVoice);
      return;
    }

    // Normal AI chat
    setChatLoading(true);
    try {
      const apiRes = await sendChatMessage(newMessages, language.name);
      const assistantMsg: ChatMessage = { role: "assistant", content: apiRes.response };
      setMessages((prev) => [...prev, assistantMsg]);
      messagesRef.current = [...newMessages, assistantMsg];
      applyCollectedFields(apiRes.collected_fields);
      speak(apiRes.response, undefined, resumeIfVoice);
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: t("sorryError") }]);
      resumeIfVoice();
    } finally {
      setChatLoading(false);
    }
  }, [chatLoading, language.name, isLanguageSelected, speak, applyCollectedFields, t, isFormValid, handleSubmit, setLanguage]);

  // Keep the ref in sync so onSilence always calls the latest sendMessage
  useEffect(() => {
    sendMessageRef.current = sendMessage;
  }, [sendMessage]);

  // Toggle voice mode
  const toggleVoice = useCallback(() => {
    if (voiceActive) {
      setVoiceActive(false);
      stopListening();
      stopSpeaking();
    } else {
      setVoiceActive(true);
      startListening();
    }
  }, [voiceActive, stopListening, stopSpeaking, startListening]);

  const updateForm = (field: keyof ComplaintFormData, value: string) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  if (submitted) {
    return (
      <div className="min-h-[calc(100vh-120px)] flex items-center justify-center px-4">
        <div className="text-center animate-reveal-up bg-card rounded-xl p-12 shadow-lg border border-border max-w-md gov-border-top">
          <CheckCircle className="w-16 h-16 text-status-resolved mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2">{t("complaintSuccess")}</h2>
          <p className="text-muted-foreground mb-4">{t("complaintRegistered")}</p>
          <div className="bg-muted rounded-lg p-4 mb-6">
            <p className="text-sm text-muted-foreground">{t("ticketId")}</p>
            <p className="text-2xl font-bold text-primary tabular-nums">{ticketId}</p>
          </div>
          <p className="text-sm text-muted-foreground">{t("saveTicketId")}</p>
        </div>
      </div>
    );
  }

  const micLabel = voiceActive
    ? isListening
      ? "🔴 Listening..."
      : isSpeaking
      ? "🔊 AI Speaking..."
      : chatLoading
      ? "⏳ Thinking..."
      : "🎙 Voice On"
    : t("speakNow");

  return (
    <div className="min-h-[calc(100vh-120px)] max-w-[1400px] mx-auto px-4 py-6">
      <h1 className="text-2xl font-bold mb-6 animate-reveal-up">{t("fileComplaint")}</h1>

      <div className="grid lg:grid-cols-[auto_1fr_1fr] gap-6">
        {/* LEFT: Mic */}
        <div className="flex lg:flex-col items-center gap-4 animate-reveal-up" style={{ animationDelay: "80ms" }}>
          <div className="flex flex-col items-center gap-2">
            <MicButton
              isListening={voiceActive}
              onToggle={toggleVoice}
              size="lg"
            />
            <p className="text-xs text-muted-foreground text-center max-w-[80px] leading-tight">
              {micLabel}
            </p>
          </div>
          {voiceActive && (
            <div className="flex flex-col items-center gap-1 mt-1">
              <div className="flex gap-0.5 items-end h-5">
                {[1,2,3,4,5].map((i) => (
                  <div
                    key={i}
                    className={`w-1 rounded-full transition-all duration-150 ${
                      isListening ? "bg-destructive" : "bg-muted-foreground/30"
                    }`}
                    style={{
                      height: isListening ? `${8 + Math.random() * 12}px` : "4px",
                      animationDelay: `${i * 80}ms`,
                    }}
                  />
                ))}
              </div>
              {interimText && (
                <p className="text-xs text-muted-foreground italic max-w-[100px] truncate" title={interimText}>
                  "{interimText}"
                </p>
              )}
            </div>
          )}
        </div>

        {/* CENTER: Chat */}
        <div className="bg-card rounded-xl shadow-sm border border-border flex flex-col h-[600px] animate-reveal-up" style={{ animationDelay: "160ms" }}>
          <div className="px-4 py-3 border-b border-border bg-muted/50 rounded-t-xl flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold">{t("aiAssistant")}</h2>
              <p className="text-xs text-muted-foreground">{t("conversationIn")} {language.nativeName}</p>
            </div>
            {voiceActive && (
              <span className="text-xs bg-destructive/10 text-destructive px-2 py-0.5 rounded-full font-medium animate-pulse">
                Voice Active
              </span>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.length === 0 && (
              <div className="flex justify-start">
                <div className="bg-muted px-4 py-3 rounded-xl rounded-bl-sm text-sm text-muted-foreground max-w-[80%]">
                  {voiceActive
                    ? "🎙 Voice mode active! Speak and I'll respond automatically."
                    : "Hello! Click the mic to use voice mode, or type your message below. How can I help you?"}
                </div>
              </div>
            )}
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-msg-in`}
              >
                <div
                  className={`max-w-[80%] px-4 py-2.5 rounded-xl text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground rounded-br-sm"
                      : "bg-muted text-foreground rounded-bl-sm"
                  }`}
                >
                  {msg.content}
                  {msg.role === "assistant" && (
                    <button
                      onClick={() => speak(msg.content)}
                      className="inline-block ml-2 text-accent hover:opacity-70"
                      aria-label="Read aloud"
                    >
                      <Volume2 size={14} />
                    </button>
                  )}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="flex justify-start">
                <div className="bg-muted px-4 py-3 rounded-xl rounded-bl-sm">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="p-3 border-t border-border">
            <div className="flex gap-2">
              <input
                type="text"
                value={interimText || inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage(inputText)}
                placeholder={voiceActive ? "🎙 Speak now — or type here too" : t("speakNow")}
                className="flex-1 px-4 py-2 rounded-lg border border-input bg-background text-foreground text-sm focus:ring-2 focus:ring-accent outline-none"
              />
              <button
                onClick={() => sendMessage(inputText)}
                disabled={(!inputText.trim() && !interimText) || chatLoading}
                className="p-2 rounded-lg bg-accent text-accent-foreground hover:opacity-90 disabled:opacity-40 active:scale-95 transition-all"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>

        {/* RIGHT: Form */}
        <div className="bg-card rounded-xl shadow-sm border border-border overflow-y-auto h-[600px] animate-reveal-up" style={{ animationDelay: "240ms" }}>
          <div className="px-4 py-3 border-b border-border bg-muted/50">
            <h2 className="text-sm font-semibold">{t("complaintForm")}</h2>
            <p className="text-xs text-muted-foreground">{t("autoFilled")}</p>
          </div>

          <div className="p-4 space-y-4">
            <FormField label={t("fullName")} value={form.fullName} onChange={(v) => updateForm("fullName", v)} />
            <FormField label={t("phone")} value={form.phone} onChange={(v) => updateForm("phone", v)} type="tel" />
            <FormField label={t("email")} value={form.email} onChange={(v) => updateForm("email", v)} type="email" />
            <FormField label={t("address")} value={form.address} onChange={(v) => updateForm("address", v)} textarea />

            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1">{t("incidentType")}</label>
              <select
                value={form.incidentType}
                onChange={(e) => updateForm("incidentType", e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-input bg-background text-foreground text-sm focus:ring-2 focus:ring-accent outline-none"
              >
                <option value="">{t("selectItem")}</option>
                {COMPLAINT_TYPES.map((type) => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </div>

            <FormField label={t("dateTime")} value={form.dateTime} onChange={(v) => updateForm("dateTime", v)} type="datetime-local" />
            <FormField label={t("description")} value={form.description} onChange={(v) => updateForm("description", v)} textarea />
            <FormField label={t("amountLost")} value={form.amountLost || ""} onChange={(v) => updateForm("amountLost", v)} placeholder="₹" />
            <FormField label={t("transactionId")} value={form.transactionId || ""} onChange={(v) => updateForm("transactionId", v)} />
            <FormField label={t("suspectDetails")} value={form.suspectDetails || ""} onChange={(v) => updateForm("suspectDetails", v)} textarea />
            <FormField label="Suspect UPI/VPA" value={form.suspectVpa || ""} onChange={(v) => updateForm("suspectVpa", v)} placeholder="e.g. suspect@upi" />
            <FormField label="Suspect Phone" value={form.suspectPhone || ""} onChange={(v) => updateForm("suspectPhone", v)} type="tel" />
            <FormField label="Suspect Bank Account" value={form.suspectBankAccount || ""} onChange={(v) => updateForm("suspectBankAccount", v)} />

            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1">{t("platform")}</label>
              <select
                value={form.platform}
                onChange={(e) => updateForm("platform", e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-input bg-background text-foreground text-sm focus:ring-2 focus:ring-accent outline-none"
              >
                <option value="">{t("selectItem")}</option>
                {PLATFORMS.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            <FileUpload
              label={t("uploadEvidence")}
              hint={t("evidenceHint")}
              files={evidenceFiles}
              onChange={setEvidenceFiles}
              accept="audio/*,video/*,.pdf,image/*"
            />
            <FileUpload
              label={t("uploadId")}
              hint={t("idHint")}
              files={idFiles}
              onChange={setIdFiles}
              accept="image/*,.pdf"
            />

            <button
              onClick={handleSubmit}
              disabled={submitting || !isFormValid}
              className="w-full py-3 rounded-lg bg-accent text-accent-foreground font-semibold hover:opacity-90 disabled:opacity-40 transition-all active:scale-[0.98]"
            >
              {submitting ? t("submitting") : t("submit")}
            </button>
            {submitError && (
              <p className="text-xs text-destructive leading-relaxed">{submitError}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Sub-components ─────────────────────────────────────────────────

function FormField({
  label, value, onChange, type = "text", textarea = false, placeholder = "",
}: {
  label: string; value: string; onChange: (v: string) => void;
  type?: string; textarea?: boolean; placeholder?: string;
}) {
  const cls = "w-full px-3 py-2 rounded-lg border border-input bg-background text-foreground text-sm focus:ring-2 focus:ring-accent outline-none transition-all";
  return (
    <div>
      <label className="block text-xs font-medium text-muted-foreground mb-1">{label}</label>
      {textarea ? (
        <textarea value={value} onChange={(e) => onChange(e.target.value)} className={`${cls} min-h-[60px] resize-y`} placeholder={placeholder} />
      ) : (
        <input type={type} value={value} onChange={(e) => onChange(e.target.value)} className={cls} placeholder={placeholder} />
      )}
    </div>
  );
}

function FileUpload({
  label, hint, files, onChange, accept,
}: {
  label: string; hint: string; files: File[]; onChange: (f: File[]) => void; accept: string;
}) {
  return (
    <div>
      <label className="block text-xs font-medium text-muted-foreground mb-1">{label}</label>
      <label className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-dashed border-input bg-muted/30 cursor-pointer hover:bg-muted/50 transition-colors">
        <Upload size={16} className="text-muted-foreground" />
        <span className="text-sm text-muted-foreground">{hint}</span>
        <input
          type="file"
          accept={accept}
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files) onChange([...files, ...Array.from(e.target.files)]);
          }}
        />
      </label>
      {files.length > 0 && (
        <div className="mt-1 space-y-0.5">
          {files.map((f, i) => (
            <p key={i} className="text-xs text-muted-foreground truncate">📎 {f.name}</p>
          ))}
        </div>
      )}
    </div>
  );
}
