import React, { createContext, useCallback, useContext, useMemo, useState } from "react";
import { FULL_TRANSLATIONS } from "../data/translations";

export interface Language {
  code: string;
  name: string;
  nativeName: string;
  speechCode: string;
}

// Language order follows user requirement.
export const LANGUAGES: Language[] = [
  { code: "hi", name: "Hindi", nativeName: "हिन्दी", speechCode: "hi-IN" },
  { code: "kok", name: "Konkani", nativeName: "कोंकणी", speechCode: "kok-IN" },
  { code: "kn", name: "Kannada", nativeName: "ಕನ್ನಡ", speechCode: "kn-IN" },
  { code: "doi", name: "Dogri", nativeName: "डोगरी", speechCode: "doi-IN" },
  { code: "brx", name: "Bodo", nativeName: "बड़ो", speechCode: "brx-IN" },
  { code: "ur", name: "Urdu", nativeName: "اردو", speechCode: "ur-IN" },
  { code: "ta", name: "Tamil", nativeName: "தமிழ்", speechCode: "ta-IN" },
  { code: "ks", name: "Kashmiri", nativeName: "कॉशुर", speechCode: "ks-IN" },
  { code: "as", name: "Assamese", nativeName: "অসমীয়া", speechCode: "as-IN" },
  { code: "bn", name: "Bengali", nativeName: "বাংলা", speechCode: "bn-IN" },
  { code: "mr", name: "Marathi", nativeName: "मराठी", speechCode: "mr-IN" },
  { code: "sd", name: "Sindhi", nativeName: "سنڌي", speechCode: "sd-IN" },
  { code: "mai", name: "Maithili", nativeName: "मैथिली", speechCode: "mai-IN" },
  { code: "pa", name: "Punjabi", nativeName: "ਪੰਜਾਬੀ", speechCode: "pa-IN" },
  { code: "ml", name: "Malayalam", nativeName: "മലയാളം", speechCode: "ml-IN" },
  { code: "mni", name: "Manipuri", nativeName: "মৈতৈলোন্", speechCode: "mni-IN" },
  { code: "te", name: "Telugu", nativeName: "తెలుగు", speechCode: "te-IN" },
  { code: "sa", name: "Sanskrit", nativeName: "संस्कृतम्", speechCode: "sa-IN" },
  { code: "ne", name: "Nepali", nativeName: "नेपाली", speechCode: "ne-NP" },
  { code: "sat", name: "Santali", nativeName: "ᱥᱟᱱᱛᱟᱲᱤ", speechCode: "sat-IN" },
  { code: "gu", name: "Gujarati", nativeName: "ગુજરાતી", speechCode: "gu-IN" },
  { code: "or", name: "Odia", nativeName: "ଓଡ଼ିଆ", speechCode: "or-IN" },
  { code: "en", name: "English", nativeName: "English", speechCode: "en-IN" },
];

function getTranslation(langCode: string, key: string): string {
  const langDict = FULL_TRANSLATIONS[langCode];
  if (langDict && langDict[key]) {
    return langDict[key];
  }
  return FULL_TRANSLATIONS.en[key] ?? key;
}

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
  isLanguageSelected: boolean;
}

const LanguageContext = createContext<LanguageContextType | null>(null);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLang] = useState<Language>(LANGUAGES.find((lang) => lang.code === "en") || LANGUAGES[0]);
  const [isLanguageSelected, setIsLanguageSelected] = useState(true);

  const setLanguage = useCallback((lang: Language) => {
    setLang(lang);
    setIsLanguageSelected(true);
  }, []);

  const t = useCallback((key: string) => getTranslation(language.code, key), [language.code]);

  const value = useMemo(
    () => ({ language, setLanguage, t, isLanguageSelected }),
    [language, setLanguage, t, isLanguageSelected]
  );

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error("useLanguage must be used within LanguageProvider");
  return ctx;
}
