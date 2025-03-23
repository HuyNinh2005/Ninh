import streamlit as st
import pathlib
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image
import img2pdf
import os
from io import BytesIO
import tempfile

# --- Các hàm xử lý ảnh ---

def order_points(pts):
    """Sắp xếp các điểm theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái"""
    try:
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect.astype('int').tolist()
    except Exception:
        return [[0, 0], [100, 0], [100, 100], [0, 100]]

def find_dest(pts):
    """Tìm các điểm đích cho phép biến đổi phối cảnh"""
    try:
        (tl, tr, br, bl) = pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        maxWidth = max(maxWidth, 100)
        maxHeight = max(maxHeight, 100)
        
        destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
        return order_points(destination_corners)
    except Exception:
        return [[0, 0], [100, 0], [100, 100], [0, 100]]

def auto_scan(image):
    """Phát hiện và căn chỉnh tài liệu tự động.
       Việc chuyển đổi đen trắng (Otsu + Offset tự động) được thực hiện sau.
    """
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        orig_img = image.copy()
        dim_limit = 1080
        max_dim = max(orig_img.shape[:2])
        if max_dim > dim_limit:
            resize_scale = dim_limit / max_dim
            img = cv2.resize(orig_img, None, fx=resize_scale, fy=resize_scale)
        else:
            img = orig_img.copy()
        # Xóa nội dung trên ảnh
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # GrabCut
        mask = np.zeros(img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect_margin = min(20, img.shape[0] // 10, img.shape[1] // 10)
        rect = (rect_margin, rect_margin, max(1, img.shape[1] - rect_margin * 2), max(1, img.shape[0] - rect_margin * 2))
        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img = img * mask2[:, :, np.newaxis]
        except Exception:
            pass
        
        # Chuyển ảnh sang ảnh xám và làm mờ
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        
        ## Phát hiện cạnh
        canny = cv2.Canny(gray, 75, 200)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        # Tìm đường viền bằng contour
        contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return orig_img
        
        # Chỉ giữ lại đường viền lớn nhất
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        doc_contour = None
        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break
        if doc_contour is None:
            if contours:
                largest_contour = contours[0]
                peri = cv2.arcLength(largest_contour, True)
                doc_contour = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
                if len(doc_contour) != 4:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    doc_contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            else:
                return orig_img
        scale_factor = orig_img.shape[1] / img.shape[1]
        doc_contour = doc_contour.reshape(-1, 2) * scale_factor
        doc_contour = doc_contour.astype(np.int32)
        if len(doc_contour) != 4:
            x, y, w, h = cv2.boundingRect(doc_contour)
            doc_contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        ordered_pts = order_points(doc_contour)
        dst_pts = find_dest(ordered_pts)
        M = cv2.getPerspectiveTransform(np.float32(ordered_pts), np.float32(dst_pts))
        warped = cv2.warpPerspective(orig_img, M, (dst_pts[2][0], dst_pts[2][1]))
        return warped
    except Exception:
        return image

def adjust_image(image, brightness=50, contrast=50, rotate=0):
    """Điều chỉnh độ sáng, tương phản và xoay ảnh."""
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        if rotate == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotate == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        beta = (brightness - 50) * 2.55
        alpha = max(0.1, contrast / 50.0)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
    except Exception:
        return image

def apply_threshold_otsu_auto_offset(image, target=128, factor=0.5):
    """
    Áp dụng ngưỡng Otsu tự động, sau đó tính offset dựa vào giá trị trung bình của ảnh.
    Công thức: offset = (target - mean_val) * factor
    Ngưỡng cuối = ngưỡng Otsu + offset.
    """
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        auto_otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mean_val = np.mean(gray)
        auto_offset = int((target - mean_val) * factor)
        final_thresh = auto_otsu_val + auto_offset
        final_thresh = max(0, min(255, final_thresh))
        _, binary = cv2.threshold(gray, final_thresh, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    except Exception:
        return image

def convert_to_pdf_bytes(image):
    """Chuyển ảnh sang dạng bytes PDF để tải xuống"""
    if image is None:
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_pil.save(tmp_path, format='PNG')
        pdf_bytes = img2pdf.convert(tmp_path)
        os.unlink(tmp_path)
        return pdf_bytes
    except Exception:
        blank_img = Image.new('RGB', (100, 100), color='white')
        byte_arr = BytesIO()
        blank_img.save(byte_arr, format='PNG')
        try:
            return img2pdf.convert(byte_arr.getvalue())
        except:
            return b''

def convert_to_image_bytes(image, format='PNG'):
    """Chuyển ảnh sang dạng bytes để tải xuống"""
    if image is None:
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        byte_arr = BytesIO()
        img_pil.save(byte_arr, format=format)
        return byte_arr.getvalue()
    except Exception:
        try:
            blank_img = Image.new('RGB', (100, 100), color='white')
            byte_arr = BytesIO()
            blank_img.save(byte_arr, format=format)
            return byte_arr.getvalue()
        except:
            return b''

# --- Giao diện người dùng ---

st.set_page_config(
    page_title="Máy Quét Tài Liệu Thông Minh",
    page_icon="📷",
    layout="wide"
)
    
# Khởi tạo biến trong session_state nếu chưa có
if 'final_image' not in st.session_state:
    st.session_state.final_image = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = "document"
    
# Phần tiêu đề chính
st.markdown("<h1 style='text-align: center;'>📷 Máy Quét Tài Liệu Thông Minh</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Tối ưu hóa kết quả quét dựa trên đặc trưng của ảnh</p>", unsafe_allow_html=True)
    
# Thanh sidebar: Upload ảnh và điều chỉnh
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>📤 Tải Ảnh</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Chọn ảnh tài liệu", type=["jpg", "png", "jpeg", "bmp"])
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>🎚️ Điều Chỉnh</h2>", unsafe_allow_html=True)
    brightness = st.slider("Độ sáng", 0, 100, 50)
    contrast = st.slider("Độ tương phản", 0, 100, 50)
    rotate = st.selectbox("Xoay ảnh", [0, 90, 180, 270])
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>⚙️ Tùy Chọn</h2>", unsafe_allow_html=True)
    scan_mode = st.radio("Chế độ quét", ["Tự động", "Thủ công"])
    
# Xử lý upload ảnh
if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if original_image is None or original_image.size == 0:
            st.stop()
        else:
            st.session_state.original_image = original_image
            st.session_state.uploaded_filename = uploaded_file.name
    except Exception:
        st.stop()
else:
    st.info("Vui lòng tải ảnh tài liệu lên từ sidebar.")
    st.stop()
    
# Hiển thị ảnh gốc và ảnh đã điều chỉnh
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Ảnh gốc")
    st.image(original_image, channels="BGR", use_column_width=True)
with col2:
    st.markdown("### Ảnh đã điều chỉnh")
    edited_image = adjust_image(original_image, brightness, contrast, rotate)
    st.image(edited_image, channels="BGR", use_column_width=True)
    
st.markdown("<hr>", unsafe_allow_html=True)
    
# Xử lý quét tài liệu
if scan_mode == "Tự động":
    if st.button("Bắt đầu Quét Tự động", key="auto_scan"):
        with st.spinner('Đang quét...'):
            warped = auto_scan(edited_image.copy())
            final_image = apply_threshold_otsu_auto_offset(warped)
            st.session_state.final_image = final_image
            st.session_state.processing_complete = True
else:
    st.markdown("### 🖱️ Vẽ 4 góc tài liệu (theo chiều kim đồng hồ)")
    st.warning("Chỉ vẽ MỘT hình tứ giác với đúng 4 điểm!")
    h, w = edited_image.shape[:2]
    aspect_ratio = w / h
    canvas_width = min(700, w)
    canvas_height = int(canvas_width / aspect_ratio)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        background_image=Image.fromarray(cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)).resize((canvas_width, canvas_height)),
        height=canvas_height,
        width=canvas_width,
        drawing_mode="polygon",
        key="canvas",
        update_streamlit=True,
    )
    
    if st.button("Xác nhận và Quét", key="manual_scan"):
        if (canvas_result.json_data is None or "objects" not in canvas_result.json_data or len(canvas_result.json_data["objects"]) == 0):
            st.stop()
        elif len(canvas_result.json_data["objects"]) > 1:
            st.stop()
        else:
            try:
                points = []
                for p in canvas_result.json_data["objects"][0]["path"]:
                    if len(p) >= 3:
                        points.append(tuple(p[1:3]))
                        
                if len(points) < 3:
                    st.stop()
                    
                else:
                    if len(points) > 4:
                        points = points[:4]
                    elif len(points) == 3:
                        p0, p1, p2 = points
                        p3 = (p0[0] + (p2[0] - p1[0]), p0[1] + (p2[1] - p1[1]))
                        points.append(p3)
                    scaled_points = [(p[0] * w / canvas_width, p[1] * h / canvas_height) for p in points]
                    with st.spinner('Đang xử lý...'):
                        ordered_pts = order_points(scaled_points)
                        dst_pts = find_dest(ordered_pts)
                        M = cv2.getPerspectiveTransform(np.float32(ordered_pts), np.float32(dst_pts))
                        warped = cv2.warpPerspective(edited_image, M, (dst_pts[2][0], dst_pts[2][1]),
                                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        final_image = apply_threshold_otsu_auto_offset(warped)
                        st.session_state.final_image = final_image
                        st.session_state.processing_complete = True
                        
            except Exception:
                st.stop()
    
    # Hiển thị kết quả quét và tải file
if st.session_state.processing_complete and st.session_state.final_image is not None:
    st.markdown("<h2 style='text-align: center;'>Kết quả Quét</h2>", unsafe_allow_html=True)
    st.image(st.session_state.final_image, channels="BGR", use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Tải file PDF",
            data=convert_to_pdf_bytes(st.session_state.final_image),
            file_name=f"{os.path.splitext(st.session_state.uploaded_filename)[0]}.pdf",
            mime="application/pdf"
        )
    with col2:
        st.download_button(
            label="Tải file PNG",
            data=convert_to_image_bytes(st.session_state.final_image),
            file_name=f"{os.path.splitext(st.session_state.uploaded_filename)[0]}.png",
            mime="image/png"
        )