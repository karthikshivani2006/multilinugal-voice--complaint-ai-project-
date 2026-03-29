import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Suspense, lazy } from "react";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { LanguageProvider } from "@/contexts/LanguageContext";
import Navbar from "@/components/Navbar";
import HomePage from "@/pages/HomePage";
import NotFound from "@/pages/NotFound";

// Lazy load non-critical pages
const LoginPage = lazy(() => import("@/pages/LoginPage"));
const ComplaintPage = lazy(() => import("@/pages/ComplaintPage"));
const TrackPage = lazy(() => import("@/pages/TrackPage"));
const AdminLoginPage = lazy(() => import("@/pages/AdminLoginPage"));
const AdminDashboard = lazy(() => import("@/pages/AdminDashboard"));
const ProfilePage = lazy(() => import("@/pages/ProfilePage"));
const AdminProfilePage = lazy(() => import("@/pages/AdminProfilePage"));

const LoadingFallback = () => <div className="min-h-screen flex items-center justify-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"/></div>;

const queryClient = new QueryClient();

function AppLayout() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <Suspense fallback={<LoadingFallback />}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/complaint" element={<ComplaintPage />} />
          <Route path="/track" element={<TrackPage />} />
          <Route path="/admin" element={<AdminLoginPage />} />
          <Route path="/admin/dashboard" element={<AdminDashboard />} />
          <Route path="/admin/profile" element={<AdminProfilePage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
      <footer className="bg-primary text-primary-foreground/60 text-xs text-center py-4 mt-8">
        © 2025 CyberGuard AI — National Cyber Crime Complaint System | Government of India
      </footer>
    </div>
  );
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <LanguageProvider>
        <BrowserRouter>
          <AppLayout />
        </BrowserRouter>
      </LanguageProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
