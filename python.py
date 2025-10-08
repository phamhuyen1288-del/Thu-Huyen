import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai.errors import APIError
from typing import Dict, Any, List

# Cần cài đặt: pip install numpy-financial python-docx
# Tuy nhiên, tôi sẽ sử dụng numpy.npv và numpy.irr (được tích hợp sẵn)
# và giả định việc đọc file Word, vì môi trường ảo có thể không có python-docx.

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh (NPV/IRR)",
    layout="wide"
)

st.title("Ứng dụng Đánh giá Phương án Kinh doanh bằng AI 🤖📊")
st.markdown("---")

# --- Khởi tạo State ---
if 'extracted_params' not in st.session_state:
    st.session_state['extracted_params'] = None
if 'cash_flow_df' not in st.session_state:
    st.session_state['cash_flow_df'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = None

# Lấy Khóa API một lần
API_KEY = st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
    st.stop() # Dừng ứng dụng nếu không có API Key
    
CLIENT = genai.Client(api_key=API_KEY)


# --- HÀM 1: Đọc nội dung File Word (.docx) ---
# NOTE: Cần cài đặt thư viện python-docx (pip install python-docx)
def read_docx_content(docx_file_obj) -> str:
    """Đọc toàn bộ nội dung văn bản từ một đối tượng file docx đã tải lên."""
    try:
        # Cần thư viện python-docx
        import docx
        document = docx.Document(docx_file_obj)
        text_content = "\n".join([paragraph.text for paragraph in document.paragraphs])
        return text_content
    except ImportError:
        st.warning("⚠️ **Lưu ý:** Để đọc file Word, bạn cần cài đặt thư viện `python-docx`. \
            Tạm thời, ứng dụng sẽ sử dụng nội dung giả định.")
        # Dữ liệu giả định để test khi không có python-docx
        return "Tóm tắt dự án: Vốn đầu tư ban đầu là 1000 triệu VND. Dự án kéo dài 5 năm. Doanh thu hàng năm ước tính 400 triệu. Chi phí hoạt động hàng năm là 150 triệu. Tỷ suất chiết khấu (WACC) là 10%. Thuế suất là 20%."
    except Exception as e:
        st.error(f"Lỗi đọc file Word: {e}")
        return ""

# --- HÀM 2: Lọc Dữ liệu bằng AI (Yêu cầu 1) ---
@st.cache_data(show_spinner="Đang yêu cầu AI lọc dữ liệu tài chính từ báo cáo...")
def extract_financial_params(document_text: str) -> Dict[str, Any]:
    """Sử dụng Gemini API để trích xuất các tham số tài chính chính ra định dạng JSON."""
    
    # 1. Định nghĩa Schema cho kết quả JSON (Cấu trúc đầu ra)
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "investment_capital": {"type": "NUMBER", "description": "Vốn đầu tư ban đầu."},
            "project_lifespan": {"type": "INTEGER", "description": "Số năm hoạt động của dự án."},
            "revenue_per_year": {"type": "NUMBER", "description": "Doanh thu trung bình hàng năm."},
            "cost_per_year": {"type": "NUMBER", "description": "Chi phí hoạt động trung bình hàng năm (không bao gồm khấu hao)."},
            "wacc_rate": {"type": "NUMBER", "description": "Tỷ suất chiết khấu (WACC) dưới dạng thập phân (ví dụ: 0.1 cho 10%)."},
            "tax_rate": {"type": "NUMBER", "description": "Thuế suất doanh nghiệp dưới dạng thập phân (ví dụ: 0.2 cho 20%)."},
        },
        "required": ["investment_capital", "project_lifespan", "revenue_per_year", "cost_per_year", "wacc_rate", "tax_rate"]
    }

    # 2. Xây dựng Prompt
    prompt = f"""
    Bạn là một chuyên gia phân tích tài chính. Hãy đọc kỹ văn bản báo cáo kinh doanh sau đây, 
    trích xuất CHÍNH XÁC các tham số tài chính được yêu cầu và trả về dưới định dạng JSON.
    Nếu không tìm thấy một tham số, hãy trả về giá trị 0.
    
    Các tham số cần trích xuất (đơn vị tiền tệ KHÔNG cần thể hiện trong giá trị số):
    1. Vốn đầu tư ban đầu (Investment Capital)
    2. Dòng đời dự án (Project Lifespan)
    3. Doanh thu trung bình hàng năm (Revenue per year)
    4. Chi phí hoạt động trung bình hàng năm (Cost per year, không bao gồm Khấu hao)
    5. Tỷ suất chiết khấu WACC (WACC rate, dưới dạng thập phân, ví dụ 0.1)
    6. Thuế suất (Tax rate, dưới dạng thập phân, ví dụ 0.2)
    
    Văn bản Báo cáo Kinh doanh:
    ---
    {document_text[:8000]} 
    ---
    """
    
    try:
        response = CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        # Parse JSON output
        import json
        return json.loads(response.text)
    
    except APIError as e:
        st.error(f"Lỗi gọi Gemini API trong quá trình trích xuất: {e}")
        return None
    except Exception as e:
        st.error(f"Lỗi không xác định trong quá trình trích xuất: {e}")
        return None

# --- HÀM 3: Xây dựng Bảng Dòng Tiền & Tính toán Chỉ số (Yêu cầu 2 & 3) ---
@st.cache_data
def calculate_project_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Tính toán Bảng Dòng tiền, NPV, IRR, PP, DPP."""
    
    # 1. Lấy tham số
    I0 = params['investment_capital']
    N = int(params['project_lifespan'])
    Rev = params['revenue_per_year']
    Cost = params['cost_per_year']
    WACC = params['wacc_rate']
    Tax = params['tax_rate']
    
    if N <= 0:
        return {"error": "Dòng đời dự án phải lớn hơn 0."}
    if WACC <= 0:
        st.warning("WACC bằng 0 hoặc âm. Sử dụng 1e-9 để tránh lỗi chia.")
        WACC = 1e-9
    
    # 2. Tính Khấu hao (Khấu hao đường thẳng)
    Depreciation = I0 / N

    # 3. Xây dựng Dòng tiền
    years = [0] + list(range(1, N + 1))
    cash_flow_data: List[Dict[str, Any]] = []
    
    cf_list: List[float] = [-I0] # Dòng tiền ban đầu (năm 0)
    
    for year in range(1, N + 1):
        # EBT (Lợi nhuận trước thuế)
        EBT = Rev - Cost - Depreciation
        
        # Thuế (Tax)
        Tax_Amount = EBT * Tax if EBT > 0 else 0
        
        # EAT (Lợi nhuận sau thuế)
        EAT = EBT - Tax_Amount
        
        # Free Cash Flow (FCF) = EAT + Khấu hao (Vì I0 đã trừ ở năm 0)
        # Giả định: Không có Working Capital, không có Salvage Value
        FCF = EAT + Depreciation
        
        # Xử lý giá trị còn lại (Salvage Value) ở năm cuối cùng (giả định 0)
        if year == N:
             FCF += 0 # Có thể thêm giá trị thanh lý nếu có
        
        cf_list.append(FCF)

        # Thêm vào bảng dữ liệu
        cash_flow_data.append({
            'Năm': year,
            'Doanh thu (Rev)': Rev,
            'Chi phí HĐ (Cost)': Cost,
            'Khấu hao (Dep)': Depreciation,
            'Lợi nhuận trước thuế (EBT)': EBT,
            'Thuế (Tax)': Tax_Amount,
            'Lợi nhuận sau thuế (EAT)': EAT,
            'Dòng tiền tự do (FCF)': FCF
        })
        
    cash_flow_df = pd.DataFrame(cash_flow_data)

    # 4. Tính toán Chỉ số
    
    # a. Net Present Value (NPV)
    # np.npv(rate, values) - Lưu ý: values là dòng tiền từ năm 1 trở đi (không bao gồm I0)
    NPV = np.npv(WACC, cf_list[1:]) + cf_list[0] 
    
    # b. Internal Rate of Return (IRR)
    # np.irr(values) - Lưu ý: values phải bao gồm cả I0
    IRR = np.irr(cf_list)
    
    # c. Payback Period (PP) và Discounted Payback Period (DPP)
    
    # Hàm tính PP và DPP
    def calculate_payback(cf_array: List[float], discount_rate: float, is_discounted: bool):
        cumulative_cf = 0
        payback_period = 0
        
        if is_discounted:
            cf_data = [cf_array[0]] + [cf / ((1 + discount_rate) ** year) for year, cf in enumerate(cf_array[1:], 1)]
        else:
            cf_data = cf_array
            
        initial_investment = abs(cf_data[0])
        
        for year in range(1, len(cf_data)):
            cumulative_cf += cf_data[year]
            if cumulative_cf >= initial_investment:
                payback_period = year - 1 + (initial_investment - (cumulative_cf - cf_data[year])) / cf_data[year]
                return payback_period
        return np.inf # Vô hạn nếu không hoàn vốn

    PP = calculate_payback(cf_list, 0, is_discounted=False)
    DPP = calculate_payback(cf_list, WACC, is_discounted=True)
    
    # 5. Lưu kết quả
    results = {
        'cash_flow_df': cash_flow_df,
        'NPV': NPV,
        'IRR': IRR,
        'PP': PP,
        'DPP': DPP,
        'I0': I0,
        'WACC': WACC,
        'N': N
    }
    st.session_state['cash_flow_df'] = cash_flow_df
    st.session_state['metrics'] = results
    
    return results

# --- HÀM 4: Phân tích kết quả bằng AI (Yêu cầu 4) ---
@st.cache_data(show_spinner="Đang yêu cầu AI phân tích và đưa ra khuyến nghị...")
def get_ai_project_analysis(metrics: Dict[str, Any], cash_flow_markdown: str) -> str:
    """Gửi các chỉ số và bảng dòng tiền đến Gemini để nhận phân tích."""
    
    # Định dạng các chỉ số
    npv_str = f"{metrics['NPV']:,.0f}"
    irr_str = f"{metrics['IRR'] * 100:.2f}%"
    pp_str = f"{metrics['PP']:.2f} năm"
    dpp_str = f"{metrics['DPP']:.2f} năm"
    wacc_str = f"{metrics['WACC'] * 100:.2f}%"
    
    prompt = f"""
    Bạn là một chuyên gia tài chính dự án cao cấp. Dựa trên các chỉ số hiệu quả dự án sau, 
    hãy đưa ra một bài phân tích và khuyến nghị CHUYÊN NGHIỆP, TOÀN DIỆN (khoảng 3-4 đoạn).
    
    Bài phân tích cần bao gồm:
    1. Đánh giá về tính khả thi của dự án dựa trên NPV (so với 0) và IRR (so với WACC = {wacc_str}).
    2. Nhận xét về rủi ro thanh khoản và thời gian hoàn vốn (PP và DPP).
    3. Đưa ra khuyến nghị cuối cùng (Chấp nhận, Từ chối, hoặc Cần Thẩm định thêm).
    
    Dữ liệu dự án:
    - Vốn đầu tư ban đầu (I0): {metrics['I0']:,.0f}
    - WACC (Tỷ suất chiết khấu): {wacc_str}
    - Dòng đời dự án: {metrics['N']} năm
    - NPV: {npv_str}
    - IRR: {irr_str}
    - Payback Period (PP): {pp_str}
    - Discounted Payback Period (DPP): {dpp_str}
    
    Bảng Dòng tiền Tự do (FCF) hàng năm:
    ---
    {cash_flow_markdown}
    ---
    """
    
    try:
        response = CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except APIError as e:
        return f"Lỗi gọi Gemini API: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# =======================================================================
# --- GIAO DIỆN STREAMLIT ---
# =======================================================================

# --- Bước 1: Tải File Word ---
st.subheader("1. Tải File Báo cáo Kinh doanh (Word)")
uploaded_file = st.file_uploader(
    "Vui lòng tải lên file Word (.docx) chứa thông tin dự án",
    type=['docx']
)

if uploaded_file:
    # --- Bước 2: Lọc Dữ liệu (Yêu cầu 1) ---
    st.subheader("2. Lọc Dữ liệu Tài chính Chính")
    
    # Nút bấm để thực hiện thao tác lọc dữ liệu
    if st.button("Lọc Dữ liệu từ AI (Bước 1)", type="primary"):
        # Đọc nội dung file Word
        with st.spinner('Đang đọc file và trích xuất nội dung...'):
            document_text = read_docx_content(uploaded_file)
        
        if document_text:
            with st.spinner('Đang gửi văn bản tới AI để trích xuất các tham số...'):
                extracted_params = extract_financial_params(document_text)
                st.session_state['extracted_params'] = extracted_params

    # Hiển thị tham số đã lọc
    if st.session_state['extracted_params']:
        params = st.session_state['extracted_params']
        st.success("✅ AI đã trích xuất thành công các tham số:")
        
        col_inv, col_life, col_rev, col_cost, col_wacc, col_tax = st.columns(6)
        
        with col_inv: st.metric("Vốn đầu tư (I0)", f"{params['investment_capital']:,.0f} VND")
        with col_life: st.metric("Dòng đời dự án (N)", f"{params['project_lifespan']:.0f} năm")
        with col_rev: st.metric("Doanh thu/năm", f"{params['revenue_per_year']:,.0f} VND")
        with col_cost: st.metric("Chi phí HĐ/năm", f"{params['cost_per_year']:,.0f} VND")
        with col_wacc: st.metric("WACC", f"{params['wacc_rate']*100:.2f}%")
        with col_tax: st.metric("Thuế suất", f"{params['tax_rate']*100:.2f}%")
        
        st.markdown("---")

        # --- Bước 3 & 4: Tính toán Dòng tiền & Chỉ số (Yêu cầu 2 & 3) ---
        st.subheader("3. Tính toán Bảng Dòng tiền và Chỉ số Đánh giá")

        if st.button("Tính toán Chỉ số Hiệu quả Dự án (Bước 2)", type="secondary"):
            if params:
                try:
                    results = calculate_project_metrics(params)
                    if 'error' in results:
                         st.error(f"Lỗi tính toán: {results['error']}")
                    else:
                        st.success("✅ Tính toán Dòng tiền và Chỉ số hoàn tất!")
                except Exception as e:
                    st.error(f"Lỗi trong quá trình tính toán tài chính: {e}")

        # Hiển thị kết quả tính toán
        if st.session_state['metrics'] and 'cash_flow_df' in st.session_state:
            metrics = st.session_state['metrics']
            cash_flow_df = st.session_state['cash_flow_df']
            
            # Bảng Dòng tiền (Yêu cầu 2)
            st.markdown("**Bảng Dòng tiền Tự do (FCF) qua các năm:**")
            st.dataframe(cash_flow_df.style.format({
                'Doanh thu (Rev)': '{:,.0f}',
                'Chi phí HĐ (Cost)': '{:,.0f}',
                'Khấu hao (Dep)': '{:,.0f}',
                'Lợi nhuận trước thuế (EBT)': '{:,.0f}',
                'Thuế (Tax)': '{:,.0f}',
                'Lợi nhuận sau thuế (EAT)': '{:,.0f}',
                'Dòng tiền tự do (FCF)': '{:,.0f}'
            }), use_container_width=True)

            # Chỉ số Đánh giá (Yêu cầu 3)
            st.markdown("**Các Chỉ số Hiệu quả Dự án:**")
            col_npv, col_irr, col_pp, col_dpp = st.columns(4)
            
            # Định dạng cho NPV:
            npv_color = "green" if metrics['NPV'] >= 0 else "red"
            npv_icon = "⬆️" if metrics['NPV'] >= 0 else "⬇️"
            
            col_npv.markdown(f"**NPV** {npv_icon}: <span style='color:{npv_color}; font-size: 1.5em;'>{metrics['NPV']:,.0f}</span> VND", unsafe_allow_html=True)
            col_irr.metric("IRR", f"{metrics['IRR']*100:.2f}%")
            col_pp.metric("Payback Period (PP)", f"{metrics['PP']:.2f} năm")
            col_dpp.metric("Discounted PP (DPP)", f"{metrics['DPP']:.2f} năm")
            
            st.markdown("---")
            
            # --- Bước 5: Phân tích AI (Yêu cầu 4) ---
            st.subheader("4. Phân tích Chỉ số Hiệu quả Dự án (AI)")
            
            if st.button("Yêu cầu AI Phân tích Dự án (Bước 3)", type="success"):
                with st.spinner("Đang gửi kết quả và chờ AI phân tích..."):
                    cf_markdown = cash_flow_df.to_markdown(index=False)
                    ai_analysis_result = get_ai_project_analysis(metrics, cf_markdown)
                    st.session_state['ai_analysis'] = ai_analysis_result

            if 'ai_analysis' in st.session_state and st.session_state['ai_analysis']:
                st.markdown("**Kết quả Phân tích & Khuyến nghị từ Gemini AI:**")
                st.info(st.session_state['ai_analysis'])

else:
    st.info("Vui lòng tải lên file Word để bắt đầu quy trình đánh giá dự án.")
