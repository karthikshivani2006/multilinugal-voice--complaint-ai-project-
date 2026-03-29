import { useState, useEffect } from "react";
import { getAdminComplaints, updateComplaintStatus, type AdminComplaint } from "@/services/api";
import { Clock, AlertCircle, CheckCircle, ChevronRight, X, FileText, User, Mail, Phone } from "lucide-react";
import { useNavigate } from "react-router-dom";

export default function AdminDashboard() {
  const navigate = useNavigate();
  const [complaints, setComplaints] = useState<AdminComplaint[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<AdminComplaint | null>(null);
  const [filter, setFilter] = useState<"all" | "pending" | "reviewing" | "resolved">("all");

  useEffect(() => {
    loadComplaints();
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      loadComplaints();
    }, 10000);
    return () => clearInterval(timer);
  }, []);

  const loadComplaints = async () => {
    setLoading(true);
    const data = await getAdminComplaints();
    setComplaints(data);
    setLoading(false);
  };

  const handleStatusUpdate = async (ticketId: string, status: "pending" | "reviewing" | "resolved") => {
    await updateComplaintStatus(ticketId, status);
    setComplaints((prev) =>
      prev.map((c) => (c.ticketId === ticketId ? { ...c, status } : c))
    );
    if (selected?.ticketId === ticketId) {
      setSelected((prev) => prev ? { ...prev, status } : null);
    }
  };

  const filtered = filter === "all" ? complaints : complaints.filter((c) => c.status === filter);

  const statusConfig = {
    pending: { icon: Clock, color: "text-status-pending", bg: "bg-status-pending/10", label: "Pending" },
    reviewing: { icon: AlertCircle, color: "text-status-reviewing", bg: "bg-status-reviewing/10", label: "Reviewing" },
    resolved: { icon: CheckCircle, color: "text-status-resolved", bg: "bg-status-resolved/10", label: "Resolved" },
  };

  const counts = {
    all: complaints.length,
    pending: complaints.filter((c) => c.status === "pending").length,
    reviewing: complaints.filter((c) => c.status === "reviewing").length,
    resolved: complaints.filter((c) => c.status === "resolved").length,
  };

  const handleLogout = () => {
    localStorage.removeItem("adminToken");
    navigate("/admin");
  };

  return (
    <div className="min-h-[calc(100vh-120px)] max-w-[1400px] mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-6 animate-reveal-up">
        <div>
          <h1 className="text-2xl font-bold">Admin Dashboard</h1>
          <p className="text-sm text-muted-foreground">CyberGuard AI — Complaint Management</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={loadComplaints}
            className="px-3 py-1.5 rounded-md border border-border text-sm hover:bg-muted/40"
          >
            Refresh
          </button>
          <button
            onClick={() => navigate("/admin/profile")}
            className="px-3 py-1.5 rounded-md border border-border text-sm hover:bg-muted/40"
          >
            Profile
          </button>
          <button
            onClick={handleLogout}
            className="px-3 py-1.5 rounded-md bg-primary text-primary-foreground text-sm hover:opacity-90"
          >
            Logout
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 animate-reveal-up" style={{ animationDelay: "80ms" }}>
        {(["all", "pending", "reviewing", "resolved"] as const).map((key) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`p-4 rounded-xl border transition-all duration-200 text-left ${
              filter === key
                ? "border-accent bg-accent/5 shadow-sm"
                : "border-border bg-card hover:border-accent/30"
            }`}
          >
            <p className="text-xs text-muted-foreground capitalize">{key === "all" ? "Total" : key}</p>
            <p className="text-2xl font-bold tabular-nums">{counts[key]}</p>
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-[1fr_400px] gap-6">
        {/* Complaints List */}
        <div className="bg-card rounded-xl shadow-sm border border-border overflow-hidden animate-reveal-up" style={{ animationDelay: "160ms" }}>
          <div className="px-4 py-3 border-b border-border bg-muted/50">
            <h2 className="text-sm font-semibold">Complaints ({filtered.length})</h2>
          </div>

          {loading ? (
            <div className="p-8 text-center text-muted-foreground">Loading...</div>
          ) : (
            <div className="divide-y divide-border">
              {filtered.map((c) => {
                const cfg = statusConfig[c.status];
                return (
                  <button
                    key={c.id}
                    onClick={() => setSelected(c)}
                    className={`w-full text-left px-4 py-3 hover:bg-muted/30 transition-colors flex items-center gap-3 ${
                      selected?.id === c.id ? "bg-muted/50" : ""
                    }`}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-mono text-muted-foreground">{c.ticketId}</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${cfg.bg} ${cfg.color} font-medium`}>
                          {cfg.label}
                        </span>
                      </div>
                      <p className="text-sm font-medium truncate">{c.fullName}</p>
                      <p className="text-xs text-muted-foreground truncate">{c.incidentType}</p>
                    </div>
                    <ChevronRight size={16} className="text-muted-foreground shrink-0" />
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Detail Panel */}
        <div className="bg-card rounded-xl shadow-sm border border-border overflow-hidden animate-reveal-up" style={{ animationDelay: "240ms" }}>
          {selected ? (
            <div className="h-full flex flex-col">
              <div className="px-4 py-3 border-b border-border bg-muted/50 flex items-center justify-between">
                <h2 className="text-sm font-semibold">Complaint Details</h2>
                <button onClick={() => setSelected(null)} className="text-muted-foreground hover:text-foreground">
                  <X size={16} />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-muted-foreground">{selected.ticketId}</span>
                  <span className="text-xs text-muted-foreground">• {selected.filedDate}</span>
                </div>

                <div className="space-y-3">
                  <DetailRow icon={User} label="Name" value={selected.fullName} />
                  <DetailRow icon={Mail} label="Email" value={selected.email} />
                  <DetailRow icon={Phone} label="Phone" value={selected.phone} />
                  <DetailRow icon={FileText} label="Type" value={selected.incidentType} />
                  {selected.dateTime && <DetailRow icon={FileText} label="Incident Date/Time" value={selected.dateTime} />}
                  {selected.platform && <DetailRow icon={FileText} label="Platform" value={selected.platform} />}
                  {selected.amountLost && <DetailRow icon={FileText} label="Amount Lost" value={selected.amountLost} />}
                  {selected.transactionId && <DetailRow icon={FileText} label="Transaction ID / UTR" value={selected.transactionId} />}
                  {selected.suspectDetails && <DetailRow icon={FileText} label="Suspect Details" value={selected.suspectDetails} />}
                  {selected.suspectVpa && <DetailRow icon={FileText} label="Suspect VPA" value={selected.suspectVpa} />}
                  {selected.suspectPhone && <DetailRow icon={Phone} label="Suspect Phone" value={selected.suspectPhone} />}
                  {selected.suspectBankAccount && <DetailRow icon={FileText} label="Suspect Bank Account" value={selected.suspectBankAccount} />}
                </div>

                <div>
                  <p className="text-xs text-muted-foreground mb-1">Description</p>
                  <p className="text-sm bg-muted/50 rounded-lg p-3">{selected.description}</p>
                </div>

                {selected.evidenceText && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Extracted Evidence Text</p>
                    <p className="text-sm bg-muted/50 rounded-lg p-3 italic">{selected.evidenceText}</p>
                  </div>
                )}

                {selected.evidenceFiles && selected.evidenceFiles.length > 0 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Evidence / Documents</p>
                    <div className="bg-muted/50 rounded-lg p-3 space-y-2">
                      {selected.evidenceFiles.map((file, idx) => (
                        <div key={`${file.filePath}-${idx}`} className="rounded-md border border-border bg-background p-2">
                          <a
                            href={file.fileUrl}
                            target="_blank"
                            rel="noreferrer"
                            className="block text-sm text-accent hover:underline break-all"
                          >
                            Open Document {idx + 1}
                          </a>
                          {file.extractedText && (
                            <p className="text-xs text-muted-foreground mt-1 whitespace-pre-wrap">
                              Extracted (English): {file.extractedText}
                            </p>
                          )}
                          {file.extractedTextOriginal && file.extractedTextOriginal !== file.extractedText && (
                            <p className="text-xs text-muted-foreground mt-1 whitespace-pre-wrap">
                              Original Extracted Text: {file.extractedTextOriginal}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Status Update */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2">Update Status</p>
                  <div className="flex gap-2">
                    {(["pending", "reviewing", "resolved"] as const).map((s) => {
                      const cfg = statusConfig[s];
                      const isActive = selected.status === s;
                      return (
                        <button
                          key={s}
                          onClick={() => handleStatusUpdate(selected.ticketId, s)}
                          className={`flex-1 py-2 rounded-lg text-xs font-semibold transition-all active:scale-95 ${
                            isActive
                              ? `${cfg.bg} ${cfg.color} ring-1 ring-current`
                              : "bg-muted text-muted-foreground hover:bg-muted/80"
                          }`}
                        >
                          {cfg.label}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-center p-8">
              <div className="text-muted-foreground">
                <FileText className="w-10 h-10 mx-auto mb-3 opacity-40" />
                <p className="text-sm">Select a complaint to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function DetailRow({ icon: Icon, label, value }: { icon: any; label: string; value: string }) {
  return (
    <div className="flex items-start gap-2">
      <Icon size={14} className="text-muted-foreground mt-0.5 shrink-0" />
      <div>
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-sm font-medium">{value}</p>
      </div>
    </div>
  );
}
