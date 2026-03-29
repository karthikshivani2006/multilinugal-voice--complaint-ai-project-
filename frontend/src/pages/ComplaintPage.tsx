import { useState, useRef, useEffect, useCallback } from "react";
import { useLanguage, LANGUAGES } from "@/contexts/LanguageContext";
import { useSpeechToText, useTextToSpeech } from "@/hooks/useSpeech";
import { sendChatMessage, submitComplaint, validateIdProof, resetChatSession, type ChatMessage, type ComplaintFormData } from "@/services/api";
import { COMPLAINT_TYPES, PLATFORMS } from "@/data/complaintTypes";
import MicButton from "@/components/MicButton";
import { Send, Upload, CheckCircle, Volume2 } from "lucide-react";

export default function ComplaintPage() {
  const { t, language, setLanguage, isLanguageSelected } = useLanguage();

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

    // Script-based detection for better voice language routing.
    if (/[\u0C00-\u0C7F]/.test(rawText)) return LANGUAGES.find((l) => l.code === "te") || null; // Telugu
    if (/[\u0B80-\u0BFF]/.test(rawText)) return LANGUAGES.find((l) => l.code === "ta") || null; // Tamil
    if (/[\u0C80-\u0CFF]/.test(rawText)) return LANGUAGES.find((l) => l.code === "kn") || null; // Kannada
    if (/[\u0D00-\u0D7F]/.test(rawText)) return LANGUAGES.find((l) => l.code === "ml") || null; // Malayalam
    if (/[\u0980-\u09FF]/.test(rawText)) return LANGUAGES.find((l) => l.code === "bn") || null; // Bengali
    if (/[\u0A80-\u0AFF]/.test(rawText)) return LANGUAGES.find((l) => l.code === "gu") || null; // Gujarati
    if (/[\u0A00-\u0A7F]/.test(rawText)) return LANGUAGES.find((l) => l.code === "pa") || null; // Punjabi
    if (/[\u0B00-\u0B7F]/.test(rawText)) return LANGUAGES.find((l) => l.code === "or") || null; // Odia
    if (/[\u0900-\u097F]/.test(rawText)) return LANGUAGES.find((l) => l.code === "hi") || null; // Devanagari

    return null;
  }, []);

  const maybeAutoSwitchLanguage = useCallback((text: string) => {
    const detected = findLanguageFromSpeech(text);
    if (!detected) return language;

    // If currently English but speech clearly maps to a different language,
    // switch immediately so STT/TTS/API stay in that language.
    if (language.code === "en" && detected.code !== "en") {
      setLanguage(detected);
      return detected;
    }
    return language;
  }, [findLanguageFromSpeech, language, setLanguage]);

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  // Always read the latest messages from a ref to avoid stale closures
  const messagesRef = useRef<ChatMessage[]>([]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  // Start a fresh backend chat session each time user opens/refreshes Complaint page.
  useEffect(() => {
    resetChatSession();
    setMessages([]);
    messagesRef.current = [];
  }, []);

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
  const [idProofExtractedText, setIdProofExtractedText] = useState("");
  const [idProofValidationNote, setIdProofValidationNote] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [ticketId, setTicketId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");

  const isHelpLikeInput = useCallback((text: string) => {
    const v = text.trim().toLowerCase();
    if (!v) return false;
    return /(don't know|dont know|not sure|no idea|explain|guide|help me|how to|naaku teliyadu|naku teliyadu|teliyadu)/i.test(v);
  }, []);

  const isStepByStepRequest = useCallback((text: string) => {
    const v = text.trim().toLowerCase();
    if (!v) return false;
    return /(step by step|one by one|ask me step|detailed way|guide me|దశలవారీగా|ఒక్కొక్కటిగా|స్టెప్ బై స్టెప్)/i.test(v);
  }, []);

  const isLanguageChangeRequest = useCallback((text: string) => {
    const v = text.trim().toLowerCase();
    if (!v) return false;
    return /(change|switch|set|select|choose).*(language)|language.*(change|switch|set|select|choose)|speak in|talk in|భాష|भाषा|ಭಾಷೆ|மொழி|ഭാഷ|زبان/.test(v);
  }, []);

  const isGreetingInput = useCallback((text: string) => {
    const v = text.trim().toLowerCase();
    if (!v) return false;
    return /\b(hi|hello|hey|hii|helo|good morning|good evening|namaste|నమస్తే|నమస్కారం)\b/.test(v);
  }, []);

  const isComplaintStartIntent = useCallback((text: string) => {
    const v = text.trim().toLowerCase();
    if (!v) return false;
    return /(complaint\s*(cheyali|cheyyali|namodu|register|file)|file\s*(a\s*)?complaint|register\s*(a\s*)?complaint|i want to file complaint|నాకు\s*ఫిర్యాదు\s*(చేయాలి|నమోదు)|ఫిర్యాదు\s*(చేయాలి|నమోదు)|ದೂರು\s*(ನೀಡಬೇಕು|ನೋಂದಣಿ)|புகார்\s*(செய்ய|பதிவு)|എനിക്ക്\s*പരാതി\s*നൽകണം)/i.test(v);
  }, []);

  const isExplainQuestion = useCallback((text: string) => {
    const v = text.trim().toLowerCase();
    if (!v) return false;
    return /(what|why|how|where|which|explain|meaning|format|example|enti|ela|ekkada|yela|ఏమి|ఎలా|ఎక్కడ|ఎందుకు|क्या|कैसे|कहाँ|क्यों|ಏನು|ಹೇಗೆ|ಎಲ್ಲಿ|ஏன்|எப்படி|எங்கே|എന്ത്|എങ്ങനെ|എവിടെ)/i.test(v) && /\?|what|why|how|where|explain|ఎలా|ఏమి|ఎక్కడ|ಏನು|ಹೇಗೆ/.test(v);
  }, []);

  const isNoEmailIntent = useCallback((text: string) => {
    const v = text.trim().toLowerCase();
    if (!v) return false;
    return /(i\s*do\s*not\s*have\s*(an\s*)?email|i\s*don't\s*have\s*(an\s*)?email|i\s*dont\s*have\s*(an\s*)?email|no\s*email|without\s*email|mail\s*id\s*ledu|email\s*ledu|email\s*nahi\s*hai|mail\s*nahi\s*hai|skip\s*email)/i.test(v);
  }, []);

  const isIdProofAutofillRequest = useCallback((text: string) => {
    const v = text.trim().toLowerCase();
    if (!v) return false;
    return /(uploaded.*(id|proof)|id proof.*(uploaded|upload)|based on (id|proof)|extract.*id|auto\s*fill.*id|fill.*from.*proof)/i.test(v);
  }, []);

  type LocalRequiredField =
    | "fullName"
    | "phone"
    | "email"
    | "address"
    | "incidentType"
    | "dateTime"
    | "description"
    | "platform"
    | "suspectDetails"
    | "amountLost"
    | "transactionId"
    | "suspectVpa"
    | "suspectPhone"
    | "suspectBankAccount";

  const financialIncidentTypes = new Set<string>([
    "UPI / Payment Fraud",
    "Credit/Debit Card Fraud",
    "Internet Banking Fraud",
    "OTP Fraud",
    "E-Commerce Fraud / Fake Delivery",
    "Loan / Insurance Fraud",
    "SIM Swap Fraud",
    "Aadhaar Misuse",
  ]);

  const baseRequiredOrder: LocalRequiredField[] = [
    "fullName",
    "phone",
    "email",
    "address",
    "incidentType",
    "dateTime",
    "description",
    "platform",
    "suspectDetails",
  ];

  const financialRequiredOrder: LocalRequiredField[] = [
    "amountLost",
    "transactionId",
    "suspectVpa",
    "suspectPhone",
    "suspectBankAccount",
  ];
  const fieldLabelKey: Record<LocalRequiredField, string> = {
    fullName: "fullName",
    phone: "phone",
    email: "email",
    address: "address",
    incidentType: "incidentType",
    dateTime: "dateTime",
    description: "description",
    platform: "platform",
    suspectDetails: "suspectDetails",
    amountLost: "amountLost",
    transactionId: "transactionId",
    suspectVpa: "suspectVpa",
    suspectPhone: "phone",
    suspectBankAccount: "suspectDetails",
  };

  const localRequiredOrder = useCallback((): LocalRequiredField[] => {
    const required = [...baseRequiredOrder];
    if (financialIncidentTypes.has((form.incidentType || "").trim())) {
      required.push(...financialRequiredOrder);
    }
    return required;
  }, [form.incidentType]);

  const nextMissingLocalField = useCallback((): LocalRequiredField | null => {
    for (const key of localRequiredOrder()) {
      const value = String(form[key] || "").trim();
      if (!value) {
        return key;
      }
      if (["email", "address", "dateTime", "platform", "amountLost", "transactionId", "suspectDetails", "suspectVpa", "suspectPhone", "suspectBankAccount"].includes(key) && /^(n\/a|na|none|skip|unknown)$/i.test(value)) {
        continue;
      }
      if (key === "phone") {
        const normalizedPhone = value.replace(/\D/g, "").slice(-10);
        if (!/^[6-9]\d{9}$/.test(normalizedPhone)) {
          return key;
        }
      }
    }
    return null;
  }, [form, localRequiredOrder]);

  const nextMissingFieldForForm = useCallback((candidateForm: ComplaintFormData): LocalRequiredField | null => {
    const required = [...baseRequiredOrder];
    if (financialIncidentTypes.has((candidateForm.incidentType || "").trim())) {
      required.push(...financialRequiredOrder);
    }

    for (const key of required) {
      const value = String(candidateForm[key] || "").trim();
      if (!value) return key;
      if (["email", "address", "dateTime", "platform", "amountLost", "transactionId", "suspectDetails", "suspectVpa", "suspectPhone", "suspectBankAccount"].includes(key) && /^(n\/a|na|none|skip|unknown)$/i.test(value)) {
        continue;
      }
      if (key === "phone") {
        const normalizedPhone = value.replace(/\D/g, "").slice(-10);
        if (!/^[6-9]\d{9}$/.test(normalizedPhone)) {
          return key;
        }
      }
    }

    return null;
  }, [baseRequiredOrder, financialIncidentTypes, financialRequiredOrder]);

  const localizedPromptForField = useCallback((field: LocalRequiredField): string => {
    if (language.code === "en") {
      const enPrompts: Record<LocalRequiredField, string> = {
        fullName: "Please share your full name.",
        phone: "Please share your 10-digit phone number.",
        email: "Please share your email address (optional but recommended).",
        address: "Please share your current address (city and state are enough).",
        incidentType: "Please tell me the type of incident.",
        dateTime: "Please share the date and time of the incident (approximate is okay).",
        description: "Please explain what happened in detail.",
        platform: "Please share the platform where this happened.",
        suspectDetails: "Please share suspect details like name, profile, URL, phone, or account info.",
        amountLost: "Please share amount lost. Enter 0 if no money was lost.",
        transactionId: "Please share transaction ID / UTR. Enter N/A if not available.",
        suspectVpa: "Please share suspect UPI/VPA ID. Enter N/A if not available.",
        suspectPhone: "Please share suspect phone number. Enter N/A if not available.",
        suspectBankAccount: "Please share suspect bank account number. Enter N/A if not available.",
      };
      return enPrompts[field];
    }

    // Universal non-English fallback using translated field labels.
    return `${t(fieldLabelKey[field])}: ${t("speakNow")}`;
  }, [language.code, t]);

  const localizedDetailedHintForField = useCallback((field: LocalRequiredField): string => {
    if (language.code === "te") {
      const teHints: Record<LocalRequiredField, string> = {
        fullName: "ఆధార్ లేదా ఐడీ ప్రూఫ్‌లో ఉన్నట్లుగా పూర్తి పేరు చెప్పండి. ఉదాహరణ: రవి కుమార్ రెడ్డి.",
        phone: "మీ క్రియాశీల 10 అంకెల మొబైల్ నంబర్ చెప్పండి. ఉదాహరణ: 9876543210.",
        email: "ఈ ఫార్మాట్‌లో చెప్పండి: name@example.com. ఉదాహరణ: ravi123@gmail.com.",
        address: "ఇంటి నంబర్, ఏరియా, సిటీ, పిన్‌కోడ్‌తో పూర్తి చిరునామా చెప్పండి.",
        incidentType: "ఉదాహరణ: UPI Fraud, Phishing, Social Media Hacking, Card Fraud.",
        dateTime: "SMS/నోటిఫికేషన్ చూసి తేదీ + సమయం చెప్పండి. ఉదాహరణ: 2026-03-29 10:30 AM.",
        description: "ఎలా మోసం జరిగిందో, ఎప్పుడు జరిగిందో, ఎంత నష్టం జరిగిందో 2-3 వాక్యాల్లో చెప్పండి.",
        platform: "ఉదాహరణ: PhonePe, Google Pay, WhatsApp, Instagram, Email.",
        suspectDetails: "అనుమానితుడి పేరు, ఫోన్, యూపీఐ, లింక్ లేదా ప్రొఫైల్ వివరాలు చెప్పండి.",
        amountLost: "నష్టం జరిగిన మొత్తం చెప్పండి. డబ్బు నష్టం లేకపోతే 0 ఇవ్వండి.",
        transactionId: "UTR/ట్రాన్సాక్షన్ ఐడి ఉంటే చెప్పండి. లేకపోతే N/A ఇవ్వండి.",
        suspectVpa: "అనుమానితుడి UPI/VPA ఐడి చెప్పండి. లేకపోతే N/A ఇవ్వండి.",
        suspectPhone: "అనుమానితుడి ఫోన్ నంబర్ ఉంటే చెప్పండి. లేకపోతే N/A ఇవ్వండి.",
        suspectBankAccount: "అనుమానితుడి బ్యాంక్ అకౌంట్ ఉంటే చెప్పండి. లేకపోతే N/A ఇవ్వండి.",
      };
      return teHints[field];
    }

    if (language.code !== "en") {
      const label = t(fieldLabelKey[field]);
      return `${label}: ${t("speakNow")}`;
    }

    const enHints: Record<LocalRequiredField, string> = {
      fullName: "Share legal full name as in ID proof. Example: Ravi Kumar Reddy.",
      phone: "Share your active 10-digit number. Example: 9876543210.",
      email: "Email is optional. Use format name@example.com, or type N/A if you do not have one.",
      address: "Address is optional. City and state are enough. Example: Bengaluru, Karnataka.",
      incidentType: "Example: UPI Fraud, Phishing, Social Media Hacking, Card Fraud.",
      dateTime: "Exact time not required. Approximate is okay. Example: 25/03/2026 around 8 PM.",
      description: "Template: I found a fake profile on [platform]. They did [action]. I request action.",
      platform: "Example: PhonePe, Google Pay, WhatsApp, Instagram, Email.",
      suspectDetails: "Share available suspect details: profile URL, username, phone, UPI, or account details.",
      amountLost: "If no amount lost, enter 0. If unsure, enter approximate amount.",
      transactionId: "Share transaction ID/UTR. If unavailable, enter N/A.",
      suspectVpa: "Share suspect UPI/VPA if available. If unavailable, enter N/A.",
      suspectPhone: "Share suspect phone number if available. If unavailable, enter N/A.",
      suspectBankAccount: "Share suspect bank account if available. If unavailable, enter N/A.",
    };
    return enHints[field];
  }, [language.code, t]);

  const detectFieldFromText = useCallback((text: string): LocalRequiredField | null => {
    const v = text.toLowerCase();
    if (/(name|full name|aadhar name|pan name|పేరు|नाम)/.test(v)) return "fullName";
    if (/(phone|mobile|number|మొబైల్|నెంబర్|नंबर)/.test(v)) return "phone";
    if (/(email|mail id|gmail|ఇమెయిల్|ईमेल)/.test(v)) return "email";
    if (/(address|city|state|చిరునామా|పట్టణం|पता)/.test(v)) return "address";
    if (/(incident|type|category|fraud type|issue type|రకం|प्रकार)/.test(v)) return "incidentType";
    if (/(date|time|when|ఎప్పుడు|తేదీ|समय|कब)/.test(v)) return "dateTime";
    if (/(description|what happened|details|వివరణ|क्या हुआ)/.test(v)) return "description";
    if (/(platform|app|website|instagram|whatsapp|ఫ్లాట్ఫామ్|ऐप)/.test(v)) return "platform";
    return null;
  }, []);

  const optionalFieldGuidanceFromText = useCallback((text: string): string | null => {
    const v = text.toLowerCase();

    if (/(amount|money lost|loss|నష్టం|राशि)/.test(v)) {
      return language.code === "te"
        ? "మొత్తం నష్టం గురించి: డబ్బు నష్టం లేకపోతే 0 ఇవ్వండి. ఖచ్చితంగా తెలియకపోతే సుమారు మొత్తాన్ని ఇవ్వండి."
        : "Amount Lost: if no money was lost, enter 0. If unsure, enter an approximate amount.";
    }
    if (/(transaction|utr|reference|ట్రాన్సాక్షన్|utr నంబర్)/.test(v)) {
      return language.code === "te"
        ? "Transaction ID / UTR లేకపోతే N/A ఇవ్వండి. బ్యాంక్ SMS లేదా యాప్ హిస్టరీలో UTR చూసి ఇవ్వచ్చు."
        : "Transaction ID / UTR: if not applicable, enter N/A. Check bank SMS/app history for UTR.";
    }
    if (/(suspect|scammer|fraudster|నిందితుడు|మోసగాడు)/.test(v)) {
      return language.code === "te"
        ? "Suspect వివరాలు: తెలిసిన పేరు/యూజర్‌నేమ్/లింక్/ఫోన్/UPI ఏదైనా ఉంటే ఇవ్వండి. లేకపోతే N/A ఇవ్వండి."
        : "Suspect Details: provide any known name/username/link/phone/UPI. If unknown, enter N/A.";
    }
    if (/(upi|vpa|bank|account|బ్యాంక్|యూపీఐ)/.test(v)) {
      return language.code === "te"
        ? "UPI/బ్యాంక్ వివరాలు పూర్తిగా తెలియకపోయినా పరవాలేదు. మీకు తెలిసినంత ఇవ్వండి, లేకపోతే N/A ఇవ్వండి."
        : "UPI/Bank fields: partial details are fine. Enter whatever you know, else N/A.";
    }
    if (/(evidence|proof|upload|screenshot|video|audio|సాక్ష్యం)/.test(v)) {
      return language.code === "te"
        ? "సాక్ష్యం అప్లోడ్: స్క్రీన్‌షాట్, వీడియో, ఆడియో, చాట్, చెల్లింపు రుజువులు జోడించండి. సాక్ష్యం లేకపోయినా ఫిర్యాదు సమర్పించవచ్చు."
        : "Evidence Upload: add screenshots, videos, audio, chat, or payment proof. You can still submit without proof, but evidence improves action chances.";
    }
    if (/(id|aadhar|aadhaar|pan|voter|driving|government card|ఐడి|ఆధార్|పాన్)/.test(v)) {
      return language.code === "te"
        ? "ID Proof: ఆధార్/పాన్/వోటర్ ఐడి/డ్రైవింగ్ లైసెన్స్ ఏదైనా ప్రభుత్వ ఐడి సరిపోతుంది."
        : "ID Proof: Aadhaar, PAN, Voter ID, or Driving License are accepted. Any government ID is acceptable.";
    }

    return null;
  }, [language.code]);

  const normalizePhoneForSubmit = useCallback((raw: string): string => {
    const digits = (raw || "").replace(/\D/g, "");
    if (digits.length === 12 && digits.startsWith("91")) return digits.slice(2);
    if (digits.length >= 10) return digits.slice(-10);
    return digits;
  }, []);

  const normalizeEmailForSubmit = useCallback((raw: string): string => {
    const text = (raw || "").trim().replace(/\s+/g, "");
    if (!text) return "N/A";
    return text;
  }, []);

  const extractFallbackEmail = useCallback((text: string): string => {
    const match = (text || "").match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i);
    return match ? match[0].trim() : "";
  }, []);

  const extractFallbackPhone = useCallback((text: string): string => {
    const digits = (text || "").replace(/\D/g, "");
    if (!digits) return "";
    const ten = digits.length >= 10 ? digits.slice(-10) : "";
    return /^[6-9]\d{9}$/.test(ten) ? ten : "";
  }, []);

  const extractFallbackName = useCallback((text: string): string => {
    const raw = (text || "").trim();
    if (!raw) return "";

    const labeled = raw.match(/(?:name|holder\s*name|full\s*name)\s*[:\-]\s*([A-Za-z][A-Za-z\s]{2,60})/i);
    if (labeled?.[1]) return labeled[1].trim();

    const firstLine = raw.split(/\r?\n/).map((x) => x.trim()).find((x) => /^[A-Za-z][A-Za-z\s]{2,60}$/.test(x));
    return firstLine || "";
  }, []);

  const extractFallbackAddress = useCallback((text: string): string => {
    const raw = (text || "").trim();
    if (!raw) return "";
    const labeled = raw.match(/(?:address|addr)\s*[:\-]\s*([^\n]{8,120})/i);
    if (labeled?.[1]) return labeled[1].trim();
    return "";
  }, []);

  const { speak, isSpeaking, stop: stopSpeaking } = useTextToSpeech();

  const handleIdFilesChange = useCallback(async (incoming: File[]) => {
    setIdFiles(incoming);
    setIdProofValidationNote("");

    if (!incoming.length) {
      setIdProofExtractedText("");
      return;
    }

    try {
      const normalizedPayload: ComplaintFormData = {
        ...form,
        phone: normalizePhoneForSubmit(form.phone),
        email: normalizeEmailForSubmit(form.email),
        amountLost: (form.amountLost || "").trim() || "0",
        transactionId: (form.transactionId || "").trim() || "N/A",
        suspectDetails: (form.suspectDetails || "").trim() || "N/A",
        suspectVpa: (form.suspectVpa || "").trim() || "N/A",
        suspectPhone: (form.suspectPhone || "").trim() || "N/A",
        suspectBankAccount: (form.suspectBankAccount || "").trim() || "N/A",
        address: (form.address || "").trim() || "N/A",
        dateTime: (form.dateTime || "").trim() || "N/A",
        platform: (form.platform || "").trim() || "N/A",
        language: language.code,
      };

      const validation = await validateIdProof(normalizedPayload, incoming[0]);
      const analysis = validation.analysis || {};
      const extractedText = (analysis.extracted_text || "").trim();
      setIdProofExtractedText(extractedText);

      const inferredName = (analysis.name || extractFallbackName(extractedText) || "").trim();
      const inferredPhone = normalizePhoneForSubmit(analysis.phone || extractFallbackPhone(extractedText) || "");
      const inferredEmail = (analysis.email || extractFallbackEmail(extractedText) || "").trim();
      const inferredAddress = (analysis.address || extractFallbackAddress(extractedText) || "").trim();

      const mergedForm: ComplaintFormData = {
        ...form,
        fullName: form.fullName || inferredName || form.fullName,
        phone: form.phone || inferredPhone || form.phone,
        email: (!form.email || form.email === "N/A") && inferredEmail ? inferredEmail : form.email,
        address: (!form.address || form.address === "N/A") && inferredAddress ? inferredAddress : form.address,
      };

      setForm((prev) => ({
        ...prev,
        fullName: mergedForm.fullName,
        phone: mergedForm.phone,
        email: mergedForm.email,
        address: mergedForm.address,
      }));

      const nextField = nextMissingFieldForForm(mergedForm);
      const filledFields: string[] = [];
      if (!form.fullName && mergedForm.fullName) filledFields.push("full name");
      if (!form.phone && mergedForm.phone) filledFields.push("phone");
      if ((!form.email || form.email === "N/A") && mergedForm.email && mergedForm.email !== "N/A") filledFields.push("email");
      if ((!form.address || form.address === "N/A") && mergedForm.address && mergedForm.address !== "N/A") filledFields.push("address");

      const intro = filledFields.length > 0
        ? (language.code === "te"
            ? `ID ప్రూఫ్ నుంచి ఆటో-ఫిల్ చేసిన వివరాలు: ${filledFields.join(", ")}.`
            : `I auto-filled these details from your ID proof: ${filledFields.join(", ")}.`)
        : (language.code === "te"
            ? "ID ప్రూఫ్‌లోనుంచి పేరు/ఫోన్/ఈమెయిల్ స్పష్టంగా దొరకలేదు. దయచేసి మిగిలిన వివరాలు చెప్పండి."
            : "I could not clearly extract name/phone/email from the ID proof. Please share the missing details.");
      const nextPrompt = nextField
        ? `${localizedPromptForField(nextField)}\n\n${localizedDetailedHintForField(nextField)}`
        : language.code === "te"
          ? "అన్ని అవసరమైన వివరాలు వచ్చాయి. ఇప్పుడు సాక్ష్యాలు జోడించి ఫిర్యాదు సమర్పించండి."
          : "All required complaint details are captured. You can now upload evidence and submit the complaint.";
      const assistantText = `${intro}\n\n${nextPrompt}`;
      const assistantMsg: ChatMessage = { role: "assistant", content: assistantText };

      if (messagesRef.current.length === 0) {
        setMessages([assistantMsg]);
        messagesRef.current = [assistantMsg];
      } else {
        setMessages((prev) => {
          const updated = [...prev, assistantMsg];
          messagesRef.current = updated;
          return updated;
        });
      }

      if (voiceActiveRef.current) {
        speak(assistantText, language.speechCode);
      }

      if (validation.mismatchFields.length > 0) {
        setIdProofValidationNote(`Proof mismatch in: ${validation.mismatchFields.join(", ")}`);
      } else {
        setIdProofValidationNote("ID proof validated and form auto-filled from extracted details.");
      }
    } catch {
      setIdProofValidationNote("ID proof analysis failed. You can still continue and submit.");
    }
  }, [form, language.code, normalizePhoneForSubmit, normalizeEmailForSubmit, nextMissingFieldForForm, localizedPromptForField, localizedDetailedHintForField, speak, language.speechCode]);

  const localizeBackendResponse = useCallback((raw: string, userText: string): string => {
    const text = (raw || "").trim();
    if (!text) return text;

    const isGenericFallback = /please continue and share your complaint details clearly\.?/i.test(text)
      || /ai service is temporarily busy/i.test(text);
    const explicitEnglishFieldPrompt = /please share your full name|please share your 10-digit phone number|please share your email address|please share your complete address|please tell me the type of incident|please share the date and time of the incident|please explain what happened in detail|please share the platform where this happened/i.test(text);

    if (!isGenericFallback && !explicitEnglishFieldPrompt) {
      return text;
    }

    const nextField = nextMissingLocalField();
    if (!nextField) {
      return language.code === "te"
        ? "అన్ని అవసరమైన వివరాలు వచ్చాయి. ఇప్పుడు సాక్ష్యాలు జోడించి ఫిర్యాదు సమర్పించండి."
        : "All required complaint details are captured. You can now add evidence and submit the complaint.";
    }

    const prompt = localizedPromptForField(nextField);
    if (isHelpLikeInput(userText)) {
      return `${prompt}\n\n${localizedDetailedHintForField(nextField)}`;
    }
    return prompt;
  }, [isHelpLikeInput, language.code, localizedDetailedHintForField, localizedPromptForField, nextMissingLocalField]);

  useEffect(() => {
    setForm((prev) => (prev.language === language.code ? prev : { ...prev, language: language.code }));
  }, [language.code]);

  // Voice mode state
  const [voiceActive, setVoiceActive] = useState(false); // is mic toggled ON?

  const normalizedPhonePreview = normalizePhoneForSubmit(form.phone);
  const emailPreview = normalizeEmailForSubmit(form.email);
  const hasRequiredCore = Boolean(
    form.fullName
    && normalizedPhonePreview
    && form.email
    && form.address
    && form.incidentType
    && form.dateTime
    && form.description
    && form.platform
    && form.suspectDetails
  );
  const phoneLooksValid = /^[6-9]\d{9}$/.test(normalizedPhonePreview);
  const emailLooksValid = emailPreview === "N/A" || /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(emailPreview);
  const needsFinancialFields = financialIncidentTypes.has((form.incidentType || "").trim());
  const hasFinancialFields = !needsFinancialFields || Boolean(
    String(form.amountLost || "").trim()
    && String(form.transactionId || "").trim()
    && String(form.suspectVpa || "").trim()
    && String(form.suspectPhone || "").trim()
    && String(form.suspectBankAccount || "").trim()
  );
  const isFormValid = Boolean(hasRequiredCore && phoneLooksValid && emailLooksValid && hasFinancialFields);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ─── Core send + voice loop ─────────────────────────────────────────
  // Forward declared as refs so the voice callbacks can always reference the latest
  const sendMessageRef = useRef<((text: string) => Promise<void>) | null>(null);

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
  } = useSpeechToText(onSilence);


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
      const normalizedPayload: ComplaintFormData = {
        ...form,
        phone: normalizePhoneForSubmit(form.phone),
        email: normalizeEmailForSubmit(form.email),
        amountLost: (form.amountLost || "").trim() || "0",
        transactionId: (form.transactionId || "").trim() || "N/A",
        suspectDetails: (form.suspectDetails || "").trim() || "N/A",
        suspectVpa: (form.suspectVpa || "").trim() || "N/A",
        suspectPhone: (form.suspectPhone || "").trim() || "N/A",
        suspectBankAccount: (form.suspectBankAccount || "").trim() || "N/A",
        address: (form.address || "").trim() || "N/A",
        dateTime: (form.dateTime || "").trim() || "N/A",
        platform: (form.platform || "").trim() || "N/A",
        language: language.code,
      };

      if (!/^[6-9]\d{9}$/.test(normalizedPayload.phone)) {
        throw new Error("Please enter a valid 10-digit mobile number.");
      }
      if (normalizedPayload.email !== "N/A" && !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(normalizedPayload.email)) {
        throw new Error("Please enter a valid email address or N/A.");
      }

      if (idFiles.length > 0) {
        const validation = await validateIdProof(normalizedPayload, idFiles[0]);
        const analysis = validation.analysis || {};

        if (!normalizedPayload.fullName && analysis.name) normalizedPayload.fullName = analysis.name;
        if ((!normalizedPayload.email || normalizedPayload.email === "N/A") && analysis.email) normalizedPayload.email = analysis.email;
        if ((!normalizedPayload.address || normalizedPayload.address === "N/A") && analysis.address) normalizedPayload.address = analysis.address;
        if ((!normalizedPayload.phone || normalizedPayload.phone.length < 10) && analysis.phone) {
          normalizedPayload.phone = normalizePhoneForSubmit(analysis.phone);
        }

        if (validation.mismatchFields.length > 0) {
          const proceed = window.confirm(
            `ID proof details do not fully match form fields (${validation.mismatchFields.join(", ")}). Do you want to proceed anyway?`
          );
          if (!proceed) {
            setSubmitting(false);
            setSubmitError("Please correct form/ID proof mismatch before submitting.");
            return;
          }
        }
      }

      setForm((prev) => ({ ...prev, ...normalizedPayload }));
      const res = await submitComplaint({ ...normalizedPayload, evidenceFiles, idProofFiles: idFiles, language: language.code });
      setTicketId(res.ticketId);
      setSubmitted(true);
      speak(`${t("yourComplaintFiled")} ${res.ticketId}`, language.speechCode);
    } catch (err) {
      const raw = err instanceof Error ? err.message : t("failedSubmit");
      const message = raw?.trim() || t("failedSubmit");
      setSubmitError(message);
      const guidance = `${t("failedSubmit")}: ${message}`;
      setMessages((prev) => [...prev, { role: "assistant", content: guidance }]);
      speak(guidance, language.speechCode);
    }
    setSubmitting(false);
  }, [form, evidenceFiles, idFiles, language.code, language.speechCode, speak, t, normalizePhoneForSubmit, normalizeEmailForSubmit]);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || chatLoading) return;

    let effectiveLanguage = maybeAutoSwitchLanguage(text);
    const currentMessages = messagesRef.current;
    const userMsg: ChatMessage = { role: "user", content: text.trim() };
    const newMessages = [...currentMessages, userMsg];
    setMessages(newMessages);
    messagesRef.current = newMessages;
    setInputText("");

    const resumeIfVoice = () => {
      if (voiceActiveRef.current) resumeListeningRef.current?.();
    };

    // Explicit language change command should work at any time, not only when current language is English.
    const requestedLanguage = findLanguageFromSpeech(text);
    if (requestedLanguage && isLanguageChangeRequest(text)) {
      setLanguage(requestedLanguage);
      setForm((prev) => ({ ...prev, language: requestedLanguage.code }));
      effectiveLanguage = requestedLanguage;
      const confirm = requestedLanguage.code === "te"
        ? "భాష మార్చబడింది. ఇప్పుడు నేను దశలవారీగా అడుగుతాను."
        : `${requestedLanguage.nativeName} ${t("languageSelected")}`;
      setMessages((prev) => [...prev, { role: "assistant", content: confirm }]);
      speak(confirm, requestedLanguage.speechCode, resumeIfVoice);
      return;
    }

    // Language selection
    if (!isLanguageSelected) {
      const matchedLang = findLanguageFromSpeech(text);
      if (matchedLang) {
        setLanguage(matchedLang);
        setChatLoading(true);
        try {
          const apiRes = await sendChatMessage(newMessages, matchedLang.name);
          const localizedReply = apiRes.response || t("greeting");
          const assistantMsg: ChatMessage = { role: "assistant", content: localizedReply };
          setMessages((prev) => [...prev, assistantMsg]);
          messagesRef.current = [...newMessages, assistantMsg];
          applyCollectedFields(apiRes.collected_fields);
          speak(localizedReply, matchedLang.speechCode, resumeIfVoice);
        } catch {
          const fallback = localizedPromptForField(nextMissingLocalField() || "fullName");
          setMessages((prev) => [...prev, { role: "assistant", content: fallback }]);
          speak(fallback, matchedLang.speechCode, resumeIfVoice);
          resumeIfVoice();
        } finally {
          setChatLoading(false);
        }
        return;
      } else {
        const reply = t("languagePrompt");
        setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
        speak(reply, "en-IN", resumeIfVoice);
        return;
      }
    }

    // Submit trigger via voice
    const submitRegex = /^(submit|file|confirm|register|submit the form|file my complaint)$/i;
    if (submitRegex.test(text.trim()) && isFormValid) {
      const reply = t("submitting");
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
      speak(reply, effectiveLanguage.speechCode, resumeIfVoice);
      handleSubmit();
      return;
    } else if (submitRegex.test(text.trim()) && !isFormValid) {
      const reply = t("failedSubmit");
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
      speak(reply, effectiveLanguage.speechCode, resumeIfVoice);
      return;
    }

    if (isGreetingInput(text) && messagesRef.current.length <= 2) {
      const greetingReply = language.code === "en"
        ? "Hello! What can I do for you today?"
        : t("greeting");
      setMessages((prev) => [...prev, { role: "assistant", content: greetingReply }]);
      speak(greetingReply, effectiveLanguage.speechCode, resumeIfVoice);
      return;
    }

    if (isComplaintStartIntent(text)) {
      const nextField = nextMissingLocalField() || "fullName";
      const guidance = `${localizedPromptForField(nextField)}\n\n${localizedDetailedHintForField(nextField)}`;
      setMessages((prev) => [...prev, { role: "assistant", content: guidance }]);
      speak(guidance, effectiveLanguage.speechCode, resumeIfVoice);
      return;
    }

    if (isIdProofAutofillRequest(text)) {
      if (!idFiles.length) {
        const reply = language.code === "te"
          ? "ముందుగా ID Proof అప్లోడ్ చేయండి. తర్వాత నేను ఆటో-ఫిల్ చేసి మిగిలిన ప్రశ్నలతో కొనసాగిస్తాను."
          : "Please upload ID proof first. Then I will auto-fill details and continue from the next missing question.";
        setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
        speak(reply, effectiveLanguage.speechCode, resumeIfVoice);
        return;
      }

      setChatLoading(true);
      try {
        await handleIdFilesChange([...idFiles]);
      } finally {
        setChatLoading(false);
      }
      return;
    }

    // Fast-path for optional email skip so user is not stuck repeating email question.
    if (nextMissingLocalField() === "email" && isNoEmailIntent(text)) {
      setForm((prev) => ({ ...prev, email: "N/A" }));
      const nextField: LocalRequiredField = "address";
      const guidance = `${localizedPromptForField(nextField)}\n\n${localizedDetailedHintForField(nextField)}`;
      setMessages((prev) => [...prev, { role: "assistant", content: guidance }]);
      speak(guidance, effectiveLanguage.speechCode, resumeIfVoice);
      return;
    }

    // Deterministic help path: user asked for step-by-step guidance.
    if (isHelpLikeInput(text) || isStepByStepRequest(text) || isExplainQuestion(text)) {
      const optionalGuidance = optionalFieldGuidanceFromText(text);
      if (optionalGuidance) {
        setMessages((prev) => [...prev, { role: "assistant", content: optionalGuidance }]);
        speak(optionalGuidance, effectiveLanguage.speechCode, resumeIfVoice);
        return;
      }

      const askedField = detectFieldFromText(text);
      const nextField = askedField || nextMissingLocalField();
      const guidance = nextField
        ? `${localizedPromptForField(nextField)}\n\n${localizedDetailedHintForField(nextField)}\n\nIf you do not know, share approximate details or enter N/A where not applicable. You can answer in your own language.`
        : language.code === "te"
        ? "మీ ప్రధాన వివరాలు వచ్చాయి. ఇప్పుడు మిగిలిన ఐచ్చిక వివరాలు లేదా సాక్ష్యాలు జోడించి ఫిర్యాదు సమర్పించండి."
        : "Your main details are complete. Add optional details/evidence and submit the complaint.";

      setMessages((prev) => [...prev, { role: "assistant", content: guidance }]);
      speak(guidance, effectiveLanguage.speechCode, resumeIfVoice);
      return;
    }

    // Normal AI chat
    setChatLoading(true);
    try {
      const apiRes = await sendChatMessage(newMessages, effectiveLanguage.name);
      const backendText = apiRes.response || localizedPromptForField(nextMissingLocalField() || "fullName");
      const localizedReply = localizeBackendResponse(backendText, text);
      const assistantMsg: ChatMessage = { role: "assistant", content: localizedReply };
      setMessages((prev) => [...prev, assistantMsg]);
      messagesRef.current = [...newMessages, assistantMsg];
      applyCollectedFields(apiRes.collected_fields);
      speak(localizedReply, effectiveLanguage.speechCode, resumeIfVoice);
    } catch {
      const nextField = nextMissingLocalField();
      const fallback = nextField
        ? `${localizedPromptForField(nextField)}\n\n${localizedDetailedHintForField(nextField)}`
        : localizedPromptForField("fullName");
      setMessages((prev) => [...prev, { role: "assistant", content: fallback }]);
      speak(fallback, effectiveLanguage.speechCode, resumeIfVoice);
      resumeIfVoice();
    } finally {
      setChatLoading(false);
    }
  }, [chatLoading, isLanguageSelected, speak, applyCollectedFields, t, isFormValid, handleSubmit, setLanguage, findLanguageFromSpeech, maybeAutoSwitchLanguage, isLanguageChangeRequest, isGreetingInput, isComplaintStartIntent, isIdProofAutofillRequest, idFiles, handleIdFilesChange, isNoEmailIntent, isHelpLikeInput, isStepByStepRequest, isExplainQuestion, detectFieldFromText, optionalFieldGuidanceFromText, nextMissingLocalField, localizedPromptForField, localizedDetailedHintForField, language.code]);

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
      if (!isLanguageSelected) {
        const prompt = t("languagePrompt");
        setMessages((prev) => [...prev, { role: "assistant", content: prompt }]);
        // Avoid STT/TTS overlap so first real user utterance is captured.
        pauseListening();
        speak(prompt, "en-IN", () => {
          if (voiceActiveRef.current) resumeListeningRef.current?.();
        });
      } else {
        startListening();
      }
    }
  }, [voiceActive, stopListening, stopSpeaking, startListening, pauseListening, isLanguageSelected, speak, t]);

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
      ? `🔴 ${t("listening")}`
      : isSpeaking
      ? `🔊 ${t("aiAssistant")}`
      : chatLoading
      ? `⏳ ${t("thinking")}`
      : `🎙 ${t("voiceOn")}`
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
                {t("voiceOn")}
              </span>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.length === 0 && (
              <div className="flex justify-start">
                <div className="bg-muted px-4 py-3 rounded-xl rounded-bl-sm text-sm text-muted-foreground max-w-[80%]">
                  {voiceActive
                    ? t("voiceOn")
                    : t("greeting")}
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
                      onClick={() => speak(msg.content, language.speechCode)}
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
                placeholder={t("speakNow")}
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
              onChange={handleIdFilesChange}
              accept="image/*,.pdf"
            />
            {idProofValidationNote && (
              <p className="text-xs text-muted-foreground leading-relaxed">{idProofValidationNote}</p>
            )}
            {idProofExtractedText && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">Extracted Proof Text</p>
                <p className="text-xs bg-muted/50 rounded-lg p-2 whitespace-pre-wrap">{idProofExtractedText}</p>
              </div>
            )}

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
            if (e.target.files) {
              onChange([...files, ...Array.from(e.target.files)]);
            }
            e.currentTarget.value = "";
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
