Tôi hiểu yêu cầu của bạn. Dưới đây là file Python hoàn chỉnh (app.py) chứa toàn bộ logic ứng dụng Đánh giá Phương án Kinh doanh Streamlit của bạn.

Bạn chỉ cần sao chép toàn bộ nội dung này và lưu thành file app.py. Đảm bảo bạn cũng đã tạo file requirements.txt và cấu hình khóa API trong Streamlit Secrets.

Python

# app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
from google import genai
from google.genai.errors import APIError

# --- Mock hoặc Tự triển khai các hàm tài chính (Vì numpy_financial không có sẵn) ---

def npv(rate, cash_flows):
    """Tính Giá trị Hiện tại Ròng (Net Present Value - NPV)."""
    # Cash flows là mảng [C0, C1, C2, ...] với C0 là vốn đầu tư (âm)
    # Rate là WACC (tỷ lệ chiết khấu)
    total_pv = 0
    for i, cash_flow in enumerate(cash_flows):
        total_pv += cash_flow / (1 + rate) ** i
    return total_pv

def irr(cash_flows):
    """Tính Tỷ suất Sinh lời Nội bộ (Internal Rate of Return - IRR)
    Sử dụng phương pháp xấp xỉ đơn giản (Bisection Method)
    """
    if len(cash_flows) < 2:
        return np.nan
        
    # Xác định giới hạn tìm kiếm
    # Thay đổi giới hạn tìm kiếm an toàn hơn
    low_rate = -0.99999 
    high_rate = 5.0 # Mở rộng giới hạn lên 500%
    
    # Số lần lặp để đạt độ chính xác
    iterations = 100
    tolerance = 1e-6
    
    # Hàm tính NPV cho một rate cụ thể
    def calculate_npv(rate, flows):
        return sum([cf / (1 + rate)**i for i, cf in enumerate(flows)])

    # Tìm IRR bằng phương pháp chia đôi (Bisection)
    for _ in range(iterations):
        mid_rate = (low_rate + high_rate) / 2
        
        # Tránh lỗi chia cho 0 nếu 1 + mid_rate = 0 (mid_rate = -1)
        if 1 + mid_rate <= 0:
            low_rate = mid_rate # Đẩy low_rate lên
            continue
        
        npv_value = calculate_npv(mid_rate, cash_flows)

        if np.abs(npv_value) < tolerance:
            return mid_rate
        elif npv_value > 0:
            low_rate = mid_rate
        else:
            high_rate = mid_rate
            
    # Nếu không hội tụ sau 100 lần lặp, trả về giá trị trung bình
    return (low_rate + high_rate) / 2

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Hiệu quả Dự án Tài chính",
    layout="wide"
)

st.title("Ứng dụng Phân tích Hiệu quả Dự án Đầu tư 📈 (Gemini AI)")
st.markdown("---")

# --- Khởi tạo State và Khóa API ---
if 'extracted_params' not in st.session_state:
    st.session_state.extracted_params = None
if 'cash_flow_df' not in st.session_state:
    st.session_state.cash_flow_df = None
if 'metrics_df' not in st.session_state:
    st.session_state.metrics_df = None
    
# Lấy API Key từ Streamlit Secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    API_KEY = ""
    st.warning("⚠️ **LƯU Ý:** Chưa cấu hình khóa API. Vui lòng thêm `GEMINI_API_KEY` vào Streamlit Secrets.")


# --- 1. Hàm AI Lọc Dữ liệu (Sử dụng Structured Output) ---

# Tắt cache để đảm bảo AI đọc lại nội dung mới khi người dùng thay đổi
def extract_parameters(document_text, api_key):
    """Sử dụng Gemini API để trích xuất các thông số tài chính vào cấu trúc JSON."""
    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình 'GEMINI_API_KEY'.")
        return None

    st.info("AI đang đọc văn bản và trích xuất thông số...")

    prompt = f"""
    Bạn là một chuyên gia phân tích tài chính. Hãy trích xuất các thông số sau từ văn bản được cung cấp bên dưới, và đảm bảo kết quả phải là một đối tượng JSON hoàn chỉnh.

    1. Vốn đầu tư (Initial_Investment)
    2. Dòng đời dự án (Project_Lifespan)
    3. Doanh thu hàng năm (Annual_Revenue)
    4. Chi phí hoạt động hàng năm (Annual_Expense)
    5. Chi phí vốn bình quân (WACC)
    6. Thuế suất (Tax_Rate)

    Yêu cầu về đơn vị và định dạng:
    - Tất cả các giá trị tiền tệ phải được chuyển về đơn vị 'tỷ đồng' (ví dụ: 30 tỷ -> 30.0). Nếu văn bản không rõ ràng, hãy ghi nhận giá trị bạn tìm thấy.
    - WACC và Thuế suất phải được chuyển về dạng thập phân (ví dụ: 13% -> 0.13, 20% -> 0.2).

    Văn bản được cung cấp:
    ---
    {document_text}
    ---
    """
    
    # Định nghĩa JSON Schema
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "Initial_Investment": {"type": "NUMBER", "description": "Vốn đầu tư ban đầu (tỷ đồng)"},
            "Project_Lifespan": {"type": "INTEGER", "description": "Dòng đời dự án (năm)"},
            "Annual_Revenue": {"type": "NUMBER", "description": "Doanh thu hàng năm (tỷ đồng)"},
            "Annual_Expense": {"type": "NUMBER", "description": "Chi phí hoạt động hàng năm (tỷ đồng)"},
            "WACC": {"type": "NUMBER", "description": "Chi phí vốn bình quân (thập phân, ví dụ: 0.13)"},
            "Tax_Rate": {"type": "NUMBER", "description": "Thuế suất (thập phân, ví dụ: 0.2)"}
        },
        "required": ["Initial_Investment", "Project_Lifespan", "Annual_Revenue", "Annual_Expense", "WACC", "Tax_Rate"]
    }

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        
        # Xử lý chuỗi JSON trả về
        result_json_str = response.text.strip().replace("```json", "").replace("```", "")
        extracted_data = json.loads(result_json_str)
        return extracted_data
        
    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi: AI trả về định dạng JSON không hợp lệ. Vui lòng thử lại hoặc điều chỉnh văn bản.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}")
        return None


# --- 2 & 3. Hàm tính toán và Xây dựng Bảng Dòng tiền ---
@st.cache_data(show_spinner=False)
def calculate_metrics(params):
    """Tính toán OCF, xây dựng bảng dòng tiền, và tính NPV, IRR, PP, DPP."""
    
    # 1. Trích xuất các tham số
    C0 = params['Initial_Investment'] # Vốn đầu tư
    L = params['Project_Lifespan'] # Vòng đời
    R = params['Annual_Revenue'] # Doanh thu
    E = params['Annual_Expense'] # Chi phí
    WACC = params['WACC'] # Chi phí vốn
    t = params['Tax_Rate'] # Thuế suất
    
    # 2. Tính toán Khấu hao hàng năm (D) (Giả định khấu hao đều - đường thẳng)
    D_a = C0 / L
    
    # 3. Tính toán Dòng tiền hoạt động ròng (OCF)
    # OCF = (EBIT - Thuế) + Khấu hao
    # OCF = (R - E - D_a) * (1 - t) + D_a
    OCF = (R - E - D_a) * (1 - t) + D_a
    
    # 4. Xây dựng Bảng Dòng tiền
    years = np.arange(L + 1)
    
    # Dòng tiền thuần (Net Cash Flow)
    NCF = [0.0] * (L + 1)
    NCF[0] = -C0 # Vốn đầu tư ban đầu (âm)
    
    for i in range(1, L + 1):
        NCF[i] = OCF
        
    df_cf = pd.DataFrame({
        'Năm': years,
        'Doanh thu (tỷ)': [0] + [R] * L,
        'Chi phí (tỷ)': [0] + [E] * L,
        'Khấu hao (tỷ)': [0] + [D_a] * L,
        'Dòng tiền thuần (NCF, tỷ)': NCF,
    })
    
    # 5. Tính toán các chỉ số Hiệu quả Dự án
    cash_flows_for_calc = np.array(NCF)
    
    # --- NPV (Giá trị hiện tại ròng) ---
    NPV_value = npv(WACC, cash_flows_for_calc)
    
    # --- IRR (Tỷ suất sinh lời nội bộ) ---
    IRR_value = irr(cash_flows_for_calc)

    # --- PP (Thời gian Hoàn vốn) ---
    cumulative_cf = np.cumsum(NCF)
    PP_value = np.nan
    for i in range(1, L + 1):
        if cumulative_cf[i] >= 0:
            # Tính toán xấp xỉ tuyến tính
            PP_value = i - 1 + abs(cumulative_cf[i-1]) / NCF[i]
            break

    # --- DPP (Thời gian Hoàn vốn có chiết khấu) ---
    discounted_cf = [NCF[0]] # C0
    for i in range(1, L + 1):
        # Chiết khấu OCF về năm hiện tại
        DCF_i = NCF[i] / (1 + WACC) ** i
        discounted_cf.append(DCF_i)
        
    cumulative_dcf = np.cumsum(discounted_cf)
    DPP_value = np.nan
    for i in range(1, L + 1):
        if cumulative_dcf[i] >= 0:
            # Tính toán xấp xỉ tuyến tính
            DPP_value = i - 1 + abs(cumulative_dcf[i-1]) / discounted_cf[i]
            break

    # 6. Tạo bảng Tóm tắt Chỉ số
    metrics_data = {
        'Chỉ số': ['NPV', 'IRR', 'Thời gian Hoàn vốn (PP)', 'Hoàn vốn Chiết khấu (DPP)'],
        'Giá trị': [NPV_value, IRR_value, PP_value, DPP_value],
        'Đơn vị': ['Tỷ đồng', '%', 'Năm', 'Năm'],
        'Tiêu chuẩn Đánh giá': [f'> 0', f'> WACC ({WACC*100:.2f}%)', f'< Dòng đời ({L} năm)', f'< Dòng đời ({L} năm)']
    }
    df_metrics = pd.DataFrame(metrics_data)

    return df_cf, df_metrics

# --- 4. Hàm AI Phân tích Hiệu quả Dự án ---

# Tắt cache để mỗi lần nhấn nút là một lần gọi AI mới
def get_ai_analysis(metrics_df, extracted_params, api_key):
    """Gửi các chỉ số hiệu quả và thông số cơ bản đến Gemini để nhận phân tích."""
    st.info("AI đang phân tích các chỉ số tài chính...")
    
    # Định dạng các thông số cơ bản
    param_str = "\n".join([f"- {k}: {v}" for k, v in extracted_params.items()])
    
    # Định dạng bảng chỉ số (cần tabulate)
    metrics_str = metrics_df.to_markdown(index=False, floatfmt=".4f")
    
    wacc_percent = extracted_params['WACC'] * 100
    lifespan = extracted_params['Project_Lifespan']
    
    prompt = f"""
    Bạn là một chuyên gia lập dự án kinh doanh cấp cao. Dựa trên các thông số dự án và chỉ số hiệu quả tài chính sau, hãy đưa ra một đánh giá chuyên nghiệp, khách quan và toàn diện về tính khả thi của dự án.

    Đánh giá cần tập trung vào:
    1. Tiêu chí NPV, IRR so với WACC ({wacc_percent:.2f}%).
    2. Thời gian hoàn vốn (PP, DPP) so với vòng đời dự án ({lifespan} năm).
    3. Kết luận về việc nên hay không nên đầu tư, dựa trên nguyên tắc quyết định đầu tư cơ bản.
    4. Gợi ý về các rủi ro hoặc điểm cần điều chỉnh (nếu cần).

    ---
    THÔNG SỐ DỰ ÁN CƠ BẢN:
    {param_str}

    CHỈ SỐ HIỆU QUẢ DỰ ÁN:
    {metrics_str}
    ---
    """
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong quá trình phân tích: {e}"

# --- Giao diện Chính ---

st.subheader("1. Nhập liệu - Dán nội dung Phương án Kinh doanh")
st.caption("Vui lòng sao chép toàn bộ nội dung từ tài liệu/đề xuất của bạn và dán vào ô bên dưới. AI sẽ tự động trích xuất các thông số.")

# Sử dụng Text Area để dán nội dung từ file Word
document_text = st.text_area(
    "Dán nội dung Phương án Kinh doanh tại đây:",
    height=300,
    # Giá trị mẫu
    value="Vốn đầu tư 30 tỷ. dự án có vòng đời trong 10 năm, bắt đầu có dòng tiền từ cuối năm thứ nhất của dự án, mỗi năm tạo ra 3,5 tỷ doanh thu, và chi phí mỗi năm là 2 tỷ, thuế suất 20%. WACC của doanh nghiệp là 13%."
)

if st.button("Lọc Dữ liệu và Phân tích Dự án 🔍", type="primary"):
    if not document_text.strip():
        st.error("Vui lòng dán nội dung phương án kinh doanh vào ô nhập liệu.")
    elif not API_KEY:
        st.error("Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng chức năng AI.")
    else:
        # Xóa cache của hàm trích xuất để đảm bảo luôn chạy lại
        # Lưu ý: calculate_metrics vẫn dùng cache để tối ưu nếu thông số không đổi
        
        with st.spinner('Đang trích xuất thông số tài chính bằng AI...'):
            extracted_params = extract_parameters(document_text, API_KEY)
            
            if extracted_params:
                st.session_state.extracted_params = extracted_params
                
                # Tính toán các chỉ số
                df_cf, df_metrics = calculate_metrics(extracted_params)
                st.session_state.cash_flow_df = df_cf
                st.session_state.metrics_df = df_metrics
                st.success("Trích xuất và Tính toán thành công!")

# --- Hiển thị kết quả ---

if st.session_state.extracted_params:
    
    # Hiển thị các thông số đã lọc
    st.markdown("---")
    st.subheader("2. Thông số Dự án đã được AI Lọc")
    
    col1, col2, col3 = st.columns(3)
    params = st.session_state.extracted_params
    
    with col1:
        st.metric("Vốn đầu tư ban đầu ($C_0$)", f"{params['Initial_Investment']:,.2f} tỷ")
        st.metric("Doanh thu Hàng năm ($R$)", f"{params['Annual_Revenue']:,.2f} tỷ")
    with col2:
        st.metric("Dòng đời Dự án ($L$)", f"{params['Project_Lifespan']} năm")
        st.metric("Chi phí Hàng năm ($E$)", f"{params['Annual_Expense']:,.2f} tỷ")
    with col3:
        st.metric("WACC ($k$)", f"{params['WACC'] * 100:.2f}%")
        st.metric("Thuế suất ($t$)", f"{params['Tax_Rate'] * 100:.0f}%")

    
    # Hiển thị Bảng Dòng tiền
    st.markdown("---")
    st.subheader("3. Bảng Dòng tiền và Tính toán OCF")
    
    # Tính toán OCF để hiển thị
    L = params['Project_Lifespan']
    C0 = params['Initial_Investment']
    D_a = C0 / L
    R = params['Annual_Revenue']
    E = params['Annual_Expense']
    t = params['Tax_Rate']
    
    OCF_calculated = (R - E - D_a) * (1 - t) + D_a
    st.info(f"Dòng tiền Hoạt động Ròng (OCF) hàng năm là: **{OCF_calculated:,.2f} tỷ VNĐ**")

    # Hiển thị Dataframe
    st.dataframe(
        st.session_state.cash_flow_df.style.format({
            'Doanh thu (tỷ)': '{:,.2f}',
            'Chi phí (tỷ)': '{:,.2f}',
            'Khấu hao (tỷ)': '{:,.2f}',
            'Dòng tiền thuần (NCF, tỷ)': '{:,.2f}',
        }),
        use_container_width=True
    )

    # Hiển thị Chỉ số Hiệu quả Dự án
    st.markdown("---")
    st.subheader("4. Các Chỉ số Đánh giá Hiệu quả Dự án (NPV, IRR, PP, DPP)")
    
    metrics_df = st.session_state.metrics_df
    
    # Format lại bảng cho đẹp và dễ đọc
    formatted_metrics_df = metrics_df.copy()
    
    # Thêm cột kết quả đánh giá nhanh
    def evaluate(row):
        value = row['Giá trị']
        wacc = params['WACC']
        lifespan = params['Project_Lifespan']
        
        if pd.isna(value): return "Chưa xác định"
        
        if row['Chỉ số'] == 'NPV':
            return "Khả thi (NPV > 0)" if value > 0 else "Không khả thi (NPV <= 0)"
        elif row['Chỉ số'] == 'IRR':
            return "Khả thi (IRR > WACC)" if value > wacc else "Không khả thi (IRR <= WACC)"
        elif row['Chỉ số'] in ['Thời gian Hoàn vốn (PP)', 'Hoàn vốn Chiết khấu (DPP)']:
            return "Khả thi (< L)" if value < lifespan else "Không khả thi (>= L)"
        return "-"

    formatted_metrics_df['Đánh giá Nhanh'] = formatted_metrics_df.apply(evaluate, axis=1)

    st.table(
        formatted_metrics_df.style.format({
            # Định dạng IRR thành %
            'Giá trị': lambda x: f'{x * 100:,.2f}' if x < 1 and x > -1 and x not in (0, np.nan) else f'{x:,.2f}', 
            'Đơn vị': lambda x: '%' if x == '%' else x,
        }).hide(axis='index')
    )
    
    # Nút bấm Yêu cầu AI Phân tích
    st.markdown("---")
    st.subheader("5. Nhận xét Chuyên sâu từ AI")

    if st.button("Yêu cầu AI Phân tích Chỉ số (NPV, IRR,...) 🧠", key="ai_analysis_button"):
        if not API_KEY:
            st.error("Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng chức năng AI.")
        else:
            with st.spinner('Đang tổng hợp dữ liệu và gửi yêu cầu phân tích...'):
                ai_result = get_ai_analysis(
                    metrics_df=st.session_state.metrics_df,
                    extracted_params=st.session_state.extracted_params,
                    api_key=API_KEY
                )
                st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                st.success(ai_result)

else:
    st.info("Vui lòng dán nội dung phương án kinh doanh vào ô nhập liệu và nhấn nút để bắt đầu phân tích.")
