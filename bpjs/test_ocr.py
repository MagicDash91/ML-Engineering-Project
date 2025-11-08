"""
Test script for OCR Claim Service
Demonstrates how to process medical receipts
"""

import base64
from ocr_claim_service import (
    extract_receipt_data,
    validate_claim,
    create_claim,
    get_claim_by_number,
    get_claim_statistics,
    format_currency
)

def load_image_as_base64(image_path: str) -> str:
    """Load image file and convert to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_receipt_ocr(image_path: str, no_kartu: str = "0001234567890"):
    """
    Test complete OCR flow:
    1. Extract receipt data
    2. Validate claim
    3. Create claim in database
    4. Show results
    """
    print("=" * 70)
    print("🔍 TESTING OCR CLAIM FLOW")
    print("=" * 70)

    # Step 1: Load image
    print("\n📸 Step 1: Loading image...")
    try:
        image_base64 = load_image_as_base64(image_path)
        print(f"   ✅ Image loaded ({len(image_base64)} characters)")
    except Exception as e:
        print(f"   ❌ Error loading image: {e}")
        return

    # Step 2: Extract data using Gemini Vision OCR
    print("\n🤖 Step 2: Extracting data with Gemini Vision OCR...")
    receipt_data = extract_receipt_data(image_base64)

    if not receipt_data:
        print("   ❌ Failed to extract data")
        return

    print("   ✅ Data extracted successfully!")
    print("\n   📋 Extracted Information:")
    print(f"      Faskes: {receipt_data.get('nama_faskes', 'N/A')}")
    print(f"      Pasien: {receipt_data.get('nama_pasien', 'N/A')}")
    print(f"      Tanggal: {receipt_data.get('tanggal_pelayanan', 'N/A')}")
    print(f"      Diagnosa: {receipt_data.get('diagnosa', 'N/A')}")
    print(f"      Jenis: {receipt_data.get('jenis_pelayanan', 'N/A')}")
    print(f"      Total Biaya: {format_currency(receipt_data.get('total_biaya', 0))}")

    if receipt_data.get('tindakan'):
        print(f"      Tindakan: {', '.join(receipt_data['tindakan'])}")
    if receipt_data.get('obat'):
        print(f"      Obat: {', '.join(receipt_data['obat'])}")

    # Step 3: Validate claim
    print("\n⚖️  Step 3: Validating claim...")
    claim_data = {
        'no_kartu': no_kartu,
        'tanggal_pelayanan': receipt_data.get('tanggal_pelayanan'),
        'nama_faskes': receipt_data.get('nama_faskes'),
        'jenis_pelayanan': receipt_data.get('jenis_pelayanan', 'RAWAT JALAN'),
        'total_biaya': receipt_data.get('total_biaya', 0),
        'dokumen': ['struk_pembayaran']  # We have the receipt
    }

    validation = validate_claim(claim_data)

    print(f"   📊 Approval Probability: {validation['approval_probability']:.0%}")
    print(f"   💰 Estimated Approval: {format_currency(validation['estimated_approval_amount'])}")

    if validation['warnings']:
        print("\n   ⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"      {warning}")

    if validation['missing_documents']:
        print("\n   📎 Missing Documents:")
        for doc in validation['missing_documents']:
            print(f"      - {doc.replace('_', ' ').title()}")

    # Step 4: Create claim in database
    print("\n💾 Step 4: Creating claim in database...")
    result = create_claim(no_kartu, receipt_data, image_base64)

    claim = result['claim']
    print(f"   ✅ Claim created successfully!")
    print(f"   🎫 Claim Number: {claim['no_klaim']}")
    print(f"   📅 Status: {claim['status_klaim']}")

    # Step 5: Show statistics
    print("\n📊 Step 5: Current Database Statistics:")
    stats = get_claim_statistics()
    print(f"   Total Claims: {stats['total_claims']}")
    print(f"   Pending Claims: {stats['pending_claims']}")
    print(f"   Avg Approval Rate: {stats['average_approval_probability']:.1%}")
    print(f"   Total Claimed: {format_currency(stats['total_claimed_amount'])}")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return claim['no_klaim']

def test_manual_data():
    """Test with manual data (no image)"""
    print("\n" + "=" * 70)
    print("🧪 TESTING WITH MANUAL DATA (No Image)")
    print("=" * 70)

    # Simulate extracted data
    receipt_data = {
        'nama_faskes': 'RSUD Kota Jakarta',
        'nama_pasien': 'John Doe',
        'tanggal_pelayanan': '2025-01-05',
        'jenis_pelayanan': 'RAWAT JALAN',
        'diagnosa': 'Demam Typhoid',
        'tindakan': ['Konsultasi Dokter Spesialis', 'Pemeriksaan Lab Darah'],
        'obat': ['Paracetamol 500mg', 'Antibiotik Ciprofloxacin'],
        'total_biaya': 850000,
        'nama_dokter': 'Dr. Ahmad Subagyo, Sp.PD',
        'no_kwitansi': 'INV2025010500123'
    }

    print("\n📋 Simulated Receipt Data:")
    for key, value in receipt_data.items():
        print(f"   {key}: {value}")

    # Create claim
    no_kartu = "0001234567890"
    result = create_claim(no_kartu, receipt_data)

    print(f"\n✅ Claim Created!")
    print(f"   No. Klaim: {result['claim']['no_klaim']}")
    print(f"   Status: {result['claim']['status_klaim']}")
    print(f"   Approval Probability: {result['validation']['approval_probability']:.0%}")

    return result['claim']['no_klaim']

if __name__ == "__main__":
    print("\n🏥 BPJS JKN - OCR Claim Service Test\n")

    # Option 1: Test with manual data
    print("Testing with manual data first...")
    manual_claim_no = test_manual_data()

    print("\n\n" + "=" * 70)
    print("📝 USAGE INSTRUCTIONS")
    print("=" * 70)
    print("\nTo test with a real medical receipt image:")
    print("1. Place your receipt image in the bpjs folder")
    print("2. Run:")
    print("   from test_ocr import test_receipt_ocr")
    print("   test_receipt_ocr('path/to/receipt.jpg')")
    print("\nExample:")
    print("   test_receipt_ocr('receipt_sample.jpg', '0001234567890')")
    print("=" * 70)
