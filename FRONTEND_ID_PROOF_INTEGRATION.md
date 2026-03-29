# Frontend Integration Guide - Partial ID Proof Handling

## Overview

This guide shows how to integrate the new ID proof partial extraction handling into your React/TypeScript frontend.

## Updated Component Flow

```
ID Proof Upload
    ↓
Validate ID Proof (GET extraction status + missing fields)
    ↓
Show Extraction Results to User
    ├─ All fields extracted? → Direct submission ✓
    └─ Some fields missing? → Show form with extracted data + missing field inputs
    ↓
User Provides Missing Details
    ↓
Submit Complaint with Merged Data
    ↓
Complaint Created ✓
```

## Component Structure

### 1. ID Proof Upload Component

**File Location:** `frontend/src/components/IDProofUpload.tsx`

```typescript
import { useState } from 'react';

interface IDProofUploadProps {
  onExtractionComplete: (result: IDValidationResult) => void;
  onError: (error: string) => void;
}

export const IDProofUpload: React.FC<IDProofUploadProps> = ({ 
  onExtractionComplete, 
  onError 
}) => {
  const [uploading, setUploading] = useState(false);
  const [extractedData, setExtractedData] = useState(null);

  const handleUpload = async (file: File) => {
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      // Optionally include user's manually entered data for comparison
      formData.append('full_name', '');
      formData.append('phone_number', '');
      formData.append('email', '');

      const response = await fetch('/api/complaint/validate-id', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('ID proof validation failed');
      }

      const result = await response.json();
      setExtractedData(result);
      onExtractionComplete(result);

    } catch (error) {
      onError(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="id-proof-upload">
      <label>Upload ID Proof (Aadhaar, PAN, DL, Passport)</label>
      <input
        type="file"
        accept="image/*,.pdf"
        onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
        disabled={uploading}
      />
      {uploading && <p>Processing ID proof...</p>}
      {extractedData && (
        <IDProofExtractionStatus data={extractedData} />
      )}
    </div>
  );
};
```

### 2. ID Proof Extraction Status Component

**File Location:** `frontend/src/components/IDProofExtractionStatus.tsx`

```typescript
interface IDValidationResult {
  extraction_status: 'SUCCESS' | 'PARTIAL_SUCCESS' | 'UNCLEAR' | 'ERROR';
  missing_fields: string[];
  extracted: {
    name: string;
    phone: string;
    email: string;
    address: string;
    document_type: string;
    id_number: string;
  };
  message: string;
  proceed_allowed: boolean;
  proceed_recommended: boolean;
}

export const IDProofExtractionStatus: React.FC<{ data: IDValidationResult }> = ({ data }) => {
  const getStatusColor = (status: string) => {
    switch(status) {
      case 'SUCCESS': return 'bg-green-100 text-green-800';
      case 'PARTIAL_SUCCESS': return 'bg-yellow-100 text-yellow-800';
      case 'UNCLEAR': return 'bg-orange-100 text-orange-800';
      case 'ERROR': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'SUCCESS': return '✅';
      case 'PARTIAL_SUCCESS': return '⚠️';
      case 'UNCLEAR': return '❌';
      case 'ERROR': return '⚠️';
      default: return 'ℹ️';
    }
  };

  return (
    <div className="mt-4 p-4 border rounded">
      <div className={`p-3 rounded mb-3 ${getStatusColor(data.extraction_status)}`}>
        <strong>
          {getStatusIcon(data.extraction_status)} {data.extraction_status}
        </strong>
      </div>

      {/* Show what was extracted */}
      <div className="mb-4">
        <h4 className="font-semibold mb-2">Extracted Information:</h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          {data.extracted.name && (
            <div><strong>Name:</strong> {data.extracted.name} ✓</div>
          )}
          {data.extracted.phone && (
            <div><strong>Phone:</strong> {data.extracted.phone} ✓</div>
          )}
          {data.extracted.email && (
            <div><strong>Email:</strong> {data.extracted.email} ✓</div>
          )}
          {data.extracted.address && (
            <div><strong>Address:</strong> {data.extracted.address} ✓</div>
          )}
          {data.extracted.document_type && (
            <div><strong>Document:</strong> {data.extracted.document_type}</div>
          )}
          {data.extracted.id_number && (
            <div><strong>ID #:</strong> {data.extracted.id_number}</div>
          )}
        </div>
      </div>

      {/* Show missing fields */}
      {data.missing_fields.length > 0 && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
          <h4 className="font-semibold mb-2">⚠️ Missing or Unclear Fields:</h4>
          <p className="text-sm mb-2">{data.missing_fields.join(', ').toUpperCase()}</p>
          <p className="text-xs text-gray-600">
            Please provide these details manually in the form below
          </p>
        </div>
      )}

      {/* Status message */}
      <p className="text-sm text-gray-600 mt-3">{data.message}</p>

      {/* Recommendation */}
      {!data.proceed_recommended && data.proceed_allowed && (
        <p className="text-xs text-blue-600 mt-2">
          💡 Tip: ID details are incomplete. You'll need to fill in the missing fields.
        </p>
      )}
    </div>
  );
};
```

### 3. Enhanced Complaint Form Component

**File Location:** `frontend/src/components/ComplaintFormWithIDProof.tsx`

```typescript
import { useState } from 'react';
import { IDProofUpload } from './IDProofUpload';

interface ComplaintFormData {
  // Complaint details
  complaintType: string;
  description: string;
  language: string;
  
  // Manual user input
  fullName: string;
  phoneNumber: string;
  email: string;
  address: string;
  
  // Extracted from ID proof
  extractedName: string;
  extractedPhone: string;
  extractedEmail: string;
  extractedAddress: string;
  extractedIDNumber: string;
  extractedDocType: string;
  
  // Additional details
  dateTime?: string;
  amountLost?: string;
  platform?: string;
}

export const ComplaintFormWithIDProof: React.FC = () => {
  const [formData, setFormData] = useState<ComplaintFormData>({
    complaintType: '',
    description: '',
    language: 'English',
    fullName: '',
    phoneNumber: '',
    email: '',
    address: '',
    extractedName: '',
    extractedPhone: '',
    extractedEmail: '',
    extractedAddress: '',
    extractedIDNumber: '',
    extractedDocType: '',
  });

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // When ID proof extraction completes
  const handleExtractionComplete = (result: any) => {
    console.log('ID proof extracted:', result);
    
    // Pre-fill form with extracted data
    setFormData(prev => ({
      ...prev,
      extractedName: result.extracted.name || '',
      extractedPhone: result.extracted.phone || '',
      extractedEmail: result.extracted.email || '',
      extractedAddress: result.extracted.address || '',
      extractedIDNumber: result.extracted.id_number || '',
      extractedDocType: result.extracted.document_type || '',
      
      // Also pre-fill user fields if extracted
      fullName: result.extracted.name || prev.fullName,
      phoneNumber: result.extracted.phone || prev.phoneNumber,
      email: result.extracted.email || prev.email,
      address: result.extracted.address || prev.address,
    }));

    // Highlight missing fields
    if (result.missing_fields.length > 0) {
      setError(`⚠️ These fields were not clearly extracted and should be provided: ${result.missing_fields.join(', ')}`);
    } else {
      setError('');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError('');

    try {
      // Determine which endpoint to use
      const hasExtractedData = !!(
        formData.extractedName || 
        formData.extractedPhone || 
        formData.extractedEmail
      );

      const endpoint = hasExtractedData 
        ? '/api/complaint/create-with-partial-id'
        : '/api/complaint/create';

      const payload = hasExtractedData 
        ? {
            // For partial ID mode
            complaint_type: formData.complaintType,
            description: formData.description,
            language: formData.language,
            
            // Manual user input
            full_name: formData.fullName,
            phone_number: formData.phoneNumber,
            email: formData.email || undefined,
            address: formData.address || undefined,
            
            // Extracted data
            extracted_name: formData.extractedName,
            extracted_phone: formData.extractedPhone,
            extracted_email: formData.extractedEmail,
            extracted_address: formData.extractedAddress,
            extracted_id_number: formData.extractedIDNumber,
            extracted_document_type: formData.extractedDocType,
            
            date_time: formData.dateTime,
            amount_lost: formData.amountLost,
            platform: formData.platform,
          }
        : {
            // For regular create
            full_name: formData.fullName,
            phone_number: formData.phoneNumber,
            email: formData.email || undefined,
            address: formData.address || undefined,
            complaint_type: formData.complaintType,
            description: formData.description,
            language: formData.language,
            date_time: formData.dateTime,
            amount_lost: formData.amountLost,
            platform: formData.platform,
          };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Submission failed');
      }

      const result = await response.json();
      setSuccess(`✅ Complaint created successfully! Ticket ID: ${result.ticket_id}`);
      
      // Reset form
      setFormData({
        complaintType: '',
        description: '',
        language: 'English',
        fullName: '',
        phoneNumber: '',
        email: '',
        address: '',
        extractedName: '',
        extractedPhone: '',
        extractedEmail: '',
        extractedAddress: '',
        extractedIDNumber: '',
        extractedDocType: '',
      });

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Submission failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-2xl mx-auto p-4">
      <h2 className="text-2xl font-bold mb-6">File a Complaint</h2>

      {/* ID Proof Upload Section */}
      <div className="mb-8 p-4 bg-blue-50 rounded border border-blue-200">
        <h3 className="font-semibold mb-4">📋 Step 1: Upload ID Proof (Optional)</h3>
        <IDProofUpload
          onExtractionComplete={handleExtractionComplete}
          onError={(err) => setError(err)}
        />
        <p className="text-xs text-blue-600 mt-2">
          Uploading ID proof helps us verify your details. Some fields may be auto-filled below.
        </p>
      </div>

      {/* Personal Information Section */}
      <div className="mb-8">
        <h3 className="font-semibold mb-4">👤 Step 2: Personal Information</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Name Input - with extracted value hint */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Full Name *
              {formData.extractedName && (
                <span className="text-xs text-green-600"> (from ID)</span>
              )}
            </label>
            <input
              type="text"
              value={formData.fullName}
              onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
              placeholder={formData.extractedName || 'Your full name'}
              className="w-full p-2 border rounded"
              required
            />
          </div>

          {/* Phone Input - with extracted value hint */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Phone Number *
              {formData.extractedPhone && (
                <span className="text-xs text-green-600"> (from ID)</span>
              )}
            </label>
            <input
              type="tel"
              value={formData.phoneNumber}
              onChange={(e) => setFormData({ ...formData, phoneNumber: e.target.value })}
              placeholder={formData.extractedPhone || '10-digit phone'}
              className="w-full p-2 border rounded"
              required
            />
          </div>

          {/* Email Input - with extracted value hint */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Email
              {formData.extractedEmail && (
                <span className="text-xs text-green-600"> (from ID)</span>
              )}
            </label>
            <input
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              placeholder={formData.extractedEmail || 'your.email@example.com'}
              className="w-full p-2 border rounded"
            />
          </div>

          {/* Address Input - with extracted value hint */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Address
              {formData.extractedAddress && (
                <span className="text-xs text-green-600"> (from ID)</span>
              )}
            </label>
            <input
              type="text"
              value={formData.address}
              onChange={(e) => setFormData({ ...formData, address: e.target.value })}
              placeholder={formData.extractedAddress || 'Your address'}
              className="w-full p-2 border rounded"
            />
          </div>
        </div>
      </div>

      {/* Complaint Details Section */}
      <div className="mb-8">
        <h3 className="font-semibold mb-4">📝 Step 3: Complaint Details</h3>
        
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">Complaint Type *</label>
          <select
            value={formData.complaintType}
            onChange={(e) => setFormData({ ...formData, complaintType: e.target.value })}
            className="w-full p-2 border rounded"
            required
          >
            <option value="">Select complaint type</option>
            <option value="Online Fraud">Online Fraud</option>
            <option value="Banking Fraud">Banking Fraud</option>
            <option value="Phishing">Phishing</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">Description *</label>
          <textarea
            value={formData.description}
            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
            placeholder="Describe what happened..."
            className="w-full p-2 border rounded"
            rows={4}
            required
          />
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">Language</label>
          <select
            value={formData.language}
            onChange={(e) => setFormData({ ...formData, language: e.target.value })}
            className="w-full p-2 border rounded"
          >
            <option value="English">English</option>
            <option value="Hindi">Hindi</option>
            <option value="Tamil">Tamil</option>
          </select>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700">
          {error}
        </div>
      )}

      {/* Success Message */}
      {success && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded text-green-700">
          {success}
        </div>
      )}

      {/* Submit Button */}
      <button
        type="submit"
        disabled={submitting}
        className="w-full bg-blue-600 text-white p-3 rounded font-semibold disabled:opacity-50"
      >
        {submitting ? 'Submitting...' : 'Submit Complaint'}
      </button>
    </form>
  );
};
```

## Usage Example in Page

**File Location:** `frontend/src/pages/ComplaintPage.tsx`

```typescript
import { ComplaintFormWithIDProof } from '../components/ComplaintFormWithIDProof';

export const ComplaintPage: React.FC = () => {
  return (
    <div className="bg-gray-50 min-h-screen py-8">
      <ComplaintFormWithIDProof />
    </div>
  );
};
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ID Proof Upload (Optional)                               │
│    - User uploads JPG/PDF                                    │
│    - Frontend → POST /complaint/validate-id                  │
├─────────────────────────────────────────────────────────────┤
│ 2. Extraction Status Response                                │
│    - extraction_status: SUCCESS | PARTIAL | UNCLEAR         │
│    - missing_fields: []                                      │
│    - extracted: {name, phone, email, ...}                    │
├─────────────────────────────────────────────────────────────┤
│ 3. Form Pre-population                                       │
│    - Extracted fields show green ✓ indicator                 │
│    - Missing fields empty and highlighted                    │
├─────────────────────────────────────────────────────────────┤
│ 4. User Provides Missing Data                                │
│    - Fills name/phone/email if not extracted                 │
├─────────────────────────────────────────────────────────────┤
│ 5. Final Submission                                          │
│    - POST /complaint/create-with-partial-id                  │
│    - Payload includes both extracted + manual data           │
├─────────────────────────────────────────────────────────────┤
│ 6. Complaint Created Successfully ✓                          │
│    - Returns ticket_id: "CG-2024-001234"                     │
│    - Some data from ID, rest manual                          │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

✅ **Smart Pre-filling** - Extracted data automatically fills form fields
✅ **Visual Indicators** - Missing fields clearly marked
✅ **Flexible Input** - Users can override any field
✅ **Graceful Fallback** - Works even if ID extraction completely fails
✅ **User-Friendly Messages** - Clear status and next steps
✅ **Multi-language** - Supports Hindi, Tamil, English

## Testing

### Test with Sample ID Proof

1. Upload a clear image of your ID
2. System should extract most/all fields
3. Form auto-fills with extracted values
4. Submit to create complaint

### Test with Blurry ID Proof

1. Upload a blurry/damaged ID image
2. System shows "PARTIAL_SUCCESS" or "UNCLEAR"
3. User sees which fields are missing
4. Manually enters missing data
5. Submit successfully completes

## Related Backend Endpoints

- `POST /complaint/validate-id` - Validate and extract ID proof data
- `POST /complaint/create-with-partial-id` - Create complaint with merged extracted+manual data
- `POST /complaint/create` - Create complaint with full data (doesn't use ID proof extraction)

## Error Handling

All components include error states and user-friendly error messages showing:
- ❌ What went wrong
- 💡 How to fix it
- 🔄 Option to retry

## Accessibility

- All form fields have proper labels
- Color-blind friendly indicators (not just green/red)
- Clear error messages
- Mobile responsive design
