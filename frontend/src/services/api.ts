const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8010";

function getFallbackUrl(url: string): string | null {
  if (url.includes(":8000")) {
    return url.replace(":8000", ":8010");
  }
  if (url.includes(":8010")) {
    return url.replace(":8010", ":8000");
  }
  return null;
}

// ─── Types ──────────────────────────────────────────────────────────

export interface ComplaintFormData {
  fullName: string;
  phone: string;
  email: string;
  address: string;
  incidentType: string;
  dateTime: string;
  description: string;
  amountLost?: string;
  transactionId?: string;
  suspectDetails?: string;
  suspectVpa?: string;
  suspectPhone?: string;
  suspectBankAccount?: string;
  platform?: string;
  evidenceFiles?: File[];
  idProofFiles?: File[];
  language: string;
}

export interface ComplaintResponse {
  ticketId: string;
  status: "pending" | "reviewing" | "resolved";
  message: string;
}

export interface TrackResponse {
  ticketId: string;
  status: "pending" | "reviewing" | "resolved";
  details: string;
  filedDate: string;
  lastUpdated: string;
}

export interface AdminComplaint {
  id: string;
  ticketId: string;
  fullName: string;
  email: string;
  phone: string;
  incidentType: string;
  description: string;
  status: "pending" | "reviewing" | "resolved";
  filedDate: string;
  language: string;
  amountLost?: string;
  transactionId?: string;
  platform?: string;
  suspectDetails?: string;
  dateTime?: string;
  suspectVpa?: string;
  suspectPhone?: string;
  suspectBankAccount?: string;
  evidenceText?: string;
  evidenceFiles?: Array<{
    filePath: string;
    fileUrl: string;
    extractedText?: string;
    extractedTextOriginal?: string;
  }>;
}

export interface IdProofValidationResult {
  mismatchFields: string[];
  proceedRecommended: boolean;
  analysis: {
    document_type?: string;
    name?: string;
    id_number?: string;
    dob?: string;
    address?: string;
    phone?: string;
    email?: string;
    confidence?: string;
    notes?: string;
    extracted_text?: string;
  };
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface BackendTrackResponse {
  ticket_id: string;
  status: "pending" | "reviewing" | "resolved";
  complaint_type: string;
  description: string;
  language: string;
  created_at: string;
}

interface BackendAdminComplaint {
  ticket_id: string;
  name: string;
  email: string;
  phone: string;
  complaint_type: string;
  description: string;
  status: "pending" | "reviewing" | "resolved";
  language: string;
  created_at: string;
  amount_lost?: string;
  transaction_id?: string;
  platform?: string;
  suspect_details?: string;
  date_time?: string;
  suspect_vpa?: string;
  suspect_phone?: string;
  suspect_bank_account?: string;
  evidence_files?: Array<{
    file_path: string;
    file_url: string;
    extracted_text?: string;
    extracted_text_original?: string;
  }>;
}

interface BackendIdValidationResult {
  mismatch_fields: string[];
  proceed_recommended: boolean;
  analysis: {
    document_type?: string;
    name?: string;
    id_number?: string;
    dob?: string;
    address?: string;
    phone?: string;
    email?: string;
    confidence?: string;
    notes?: string;
    extracted_text?: string;
  };
}

function getSessionId() {
  const key = "cyberguard_session_id";
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const generated = `sess-${crypto.randomUUID()}`;
  localStorage.setItem(key, generated);
  return generated;
}

export function resetChatSession() {
  const key = "cyberguard_session_id";
  localStorage.setItem(key, `sess-${crypto.randomUUID()}`);
}

function adminAuthHeaders() {
  const token = localStorage.getItem("adminToken") || "";
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const hasFormDataBody = options?.body instanceof FormData;
  const headers: Record<string, string> = {
    ...(options?.headers as Record<string, string> | undefined),
  };

  if (!hasFormDataBody) {
    headers["Content-Type"] = "application/json";
  }

  const doFetch = (targetUrl: string) =>
    fetch(targetUrl, {
      ...options,
      headers,
    });

  let response: Response;
  try {
    response = await doFetch(url);
  } catch (primaryError) {
    const fallbackUrl = getFallbackUrl(url);
    if (!fallbackUrl) {
      throw primaryError;
    }
    response = await doFetch(fallbackUrl);
  }

  if (!response.ok) {
    const body = await response.text();
    let message = body || `HTTP ${response.status}`;
    try {
      const parsed = JSON.parse(body);
      if (Array.isArray(parsed?.detail)) {
        message = parsed.detail
          .map((item: any) => {
            if (typeof item === "string") return item;
            if (item && typeof item.msg === "string") {
              const loc = Array.isArray(item.loc) ? item.loc.join(".") : "field";
              return `${loc}: ${item.msg}`;
            }
            return "Validation error";
          })
          .join("; ");
      } else if (typeof parsed?.detail === "string") {
        message = parsed.detail;
      }
    } catch {
      // Keep raw response body when it is not JSON.
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

// ─── API Functions ───────────────────────────────────────────────────

/** Submit a new complaint. Returns ticket ID. */
export async function submitComplaint(data: ComplaintFormData): Promise<ComplaintResponse> {
  const payload = {
    full_name: data.fullName,
    phone_number: data.phone,
    email: !data.email || ["n/a", "na", "none", "skip"].includes(data.email.trim().toLowerCase()) ? null : data.email,
    address: data.address,
    complaint_type: data.incidentType,
    date_time: data.dateTime,
    description: data.description,
    amount_lost: data.amountLost || null,
    transaction_id: data.transactionId || null,
    platform: data.platform || null,
    suspect_details: data.suspectDetails || null,
    suspect_vpa: data.suspectVpa || null,
    suspect_phone: data.suspectPhone || null,
    suspect_bank_account: data.suspectBankAccount || null,
    language: data.language,
  };

  const created = await fetchJson<{ ticket_id: string; status: ComplaintResponse["status"]; message: string }>(
    `${BASE_URL}/complaint/create`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    }
  );

  const uploadSingleFile = async (file: File, fileKind: "evidence" | "id_proof") => {
    const form = new FormData();
    form.append("ticket_id", created.ticket_id);
    form.append("file_kind", fileKind);
    form.append("file", file);

    await fetchJson(`${BASE_URL}/complaint/upload`, {
      method: "POST",
      body: form,
    });
  };

  for (const file of data.evidenceFiles || []) {
    await uploadSingleFile(file, "evidence");
  }

  for (const file of data.idProofFiles || []) {
    await uploadSingleFile(file, "id_proof");
  }

  return {
    ticketId: created.ticket_id,
    status: created.status,
    message: created.message,
  };
}

export async function validateIdProof(data: ComplaintFormData, file: File): Promise<IdProofValidationResult> {
  const form = new FormData();
  form.append("full_name", data.fullName || "");
  form.append("phone_number", data.phone || "");
  form.append("email", data.email || "");
  form.append("address", data.address || "");
  form.append("file", file);

  const res = await fetchJson<BackendIdValidationResult>(`${BASE_URL}/complaint/validate-id`, {
    method: "POST",
    body: form,
  });

  return {
    mismatchFields: res.mismatch_fields || [],
    proceedRecommended: !!res.proceed_recommended,
    analysis: res.analysis || {},
  };
}

/** Track complaint by ticket ID */
export async function trackComplaint(ticketId: string): Promise<TrackResponse | null> {
  if (!ticketId) return null;
  try {
    const response = await fetchJson<BackendTrackResponse>(`${BASE_URL}/complaint/${ticketId}`);
    return {
      ticketId: response.ticket_id,
      status: response.status,
      details: response.description,
      filedDate: new Date(response.created_at).toLocaleDateString(),
      lastUpdated: new Date(response.created_at).toLocaleDateString(),
    };
  } catch {
    return null;
  }
}

/** User login */
export async function loginUser(creds: LoginCredentials): Promise<{ success: boolean; token?: string }> {
  if (!creds.email || !creds.password) return { success: false };
  return { success: true, token: "user-local-token" };
}

/** Admin login */
export async function loginAdmin(creds: LoginCredentials): Promise<{ success: boolean; token?: string }> {
  try {
    const response = await fetchJson<{ access_token: string }>(`${BASE_URL}/admin/login`, {
      method: "POST",
      body: JSON.stringify(creds),
    });

    if (response.access_token) {
      localStorage.setItem("adminToken", response.access_token);
      return { success: true, token: response.access_token };
    }
  } catch {
    return { success: false };
  }

  return { success: false };
}

/** Get all complaints (admin) */
export async function getAdminComplaints(): Promise<AdminComplaint[]> {
  const token = localStorage.getItem("adminToken");
  if (!token) {
    return [];
  }

  try {
    const response = await fetchJson<BackendAdminComplaint[]>(`${BASE_URL}/admin/complaints`, {
      method: "GET",
      headers: {
        ...adminAuthHeaders(),
      },
    });

    return response.map((item, idx) => ({
      id: String(idx + 1),
      ticketId: item.ticket_id,
      fullName: item.name,
      email: item.email,
      phone: item.phone,
      incidentType: item.complaint_type,
      description: item.description,
      status: item.status,
      filedDate: new Date(item.created_at).toLocaleDateString(),
      language: item.language,
      amountLost: item.amount_lost,
      transactionId: item.transaction_id,
      platform: item.platform,
      suspectDetails: item.suspect_details,
      dateTime: item.date_time,
      suspectVpa: item.suspect_vpa,
      suspectPhone: item.suspect_phone,
      suspectBankAccount: item.suspect_bank_account,
      evidenceFiles: (item.evidence_files || []).map((f) => ({
        filePath: f.file_path,
        fileUrl: f.file_url.startsWith("http") ? f.file_url : `${BASE_URL}${f.file_url.startsWith("/") ? "" : "/"}${f.file_url}`,
        extractedText: f.extracted_text,
        extractedTextOriginal: f.extracted_text_original,
      })),
      evidenceText: (item.evidence_files || []).map((f) => f.extracted_text).filter(Boolean).join("\n\n"),
    }));
  } catch {
    return [];
  }
}

/** Update complaint status (admin) */
export async function updateComplaintStatus(
  ticketId: string,
  status: "pending" | "reviewing" | "resolved"
): Promise<{ success: boolean }> {
  try {
    await fetchJson<{ message: string }>(`${BASE_URL}/admin/update-status`, {
      method: "PUT",
      headers: {
        ...adminAuthHeaders(),
      },
      body: JSON.stringify({
        ticket_id: ticketId,
        status,
      }),
    });
    return { success: true };
  } catch {
    return { success: false };
  }
}

/** Send chat message to AI and get response */
export async function sendChatMessage(
  messages: ChatMessage[],
  language: string
): Promise<{ response: string; collected_fields?: Record<string, string> }> {
  const lastMsg = messages[messages.length - 1]?.content || "";

  // Keep timeout short for responsive voice UX.
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 35_000);

  try {
    const response = await fetchJson<{ response: string; collected_fields: Record<string, string> }>(`${BASE_URL}/chat`, {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify({
        session_id: getSessionId(),
        language,
        message: lastMsg,
      }),
    });
    clearTimeout(timeoutId);
    return { response: response.response, collected_fields: response.collected_fields };
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err?.name === "AbortError") {
      return { response: "" };
    }
    return { response: "" };
  }
}

