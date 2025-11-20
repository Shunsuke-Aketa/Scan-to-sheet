"""
共通ユーティリティ関数
PDF変換、画像処理などの共通機能を提供
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import io
from PIL import Image
import platform
import os

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def get_tesseract_path() -> Optional[str]:
    """
    Tesseract OCRのパスを取得
    
    Returns:
        Tesseract実行ファイルのパス、見つからない場合はNone
    """
    if platform.system() == 'Windows':
        possible_paths = [
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
    return None


def get_tessdata_path(tesseract_path: Optional[str] = None) -> Optional[str]:
    """
    Tesseractのtessdataディレクトリのパスを取得
    
    Args:
        tesseract_path: Tesseract実行ファイルのパス（Noneの場合は自動検索）
    
    Returns:
        tessdataディレクトリのパス、見つからない場合はNone
    """
    if tesseract_path is None:
        tesseract_path = get_tesseract_path()
    
    if tesseract_path is None:
        return None
    
    # Tesseract実行ファイルのディレクトリからtessdataディレクトリを推測
    tesseract_dir = os.path.dirname(tesseract_path)
    tessdata_path = os.path.join(tesseract_dir, 'tessdata')
    
    if os.path.exists(tessdata_path) and os.path.isdir(tessdata_path):
        return tessdata_path
    
    return None


def check_japanese_data(tessdata_path: Optional[str] = None) -> bool:
    """
    日本語データファイル（jpn.traineddata）の存在を確認
    
    Args:
        tessdata_path: tessdataディレクトリのパス（Noneの場合は自動検索）
    
    Returns:
        日本語データファイルが存在する場合はTrue、存在しない場合はFalse
    """
    if tessdata_path is None:
        tessdata_path = get_tessdata_path()
    
    if tessdata_path is None:
        return False
    
    jpn_data_path = os.path.join(tessdata_path, 'jpn.traineddata')
    return os.path.exists(jpn_data_path) and os.path.isfile(jpn_data_path)


def pdf_to_images_from_path(pdf_path: str, dpi: int = 200, pages: Optional[List[int]] = None) -> List[np.ndarray]:
    """
    PDFファイルを画像に変換（画像レイヤーから）
    pdf2imageが失敗した場合、PyMuPDFをフォールバックとして使用
    
    Args:
        pdf_path: PDFファイルのパス
        dpi: 解像度（デフォルト200）
        pages: 処理するページ番号のリスト（Noneの場合は全ページ）
    
    Returns:
        画像のリスト（numpy配列）
    """
    # まずpdf2imageを試行
    if PDF2IMAGE_AVAILABLE:
        try:
            if pages is None:
                images = convert_from_path(pdf_path, dpi=dpi)
            else:
                images = convert_from_path(pdf_path, dpi=dpi, first_page=min(pages), last_page=max(pages))
                # 指定されたページのみを抽出
                if len(pages) < len(images):
                    images = [images[i - min(pages)] for i in pages if i - min(pages) < len(images)]
            
            # PIL Imageをnumpy配列に変換
            result = []
            for img in images:
                img_array = np.array(img)
                # RGBからBGRに変換（OpenCV用）
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                result.append(img_array)
            
            return result
        except Exception as e:
            # pdf2imageが失敗した場合（popplerがない場合など）、PyMuPDFをフォールバックとして使用
            if PYMUPDF_AVAILABLE:
                return pdf_to_images_with_pymupdf(pdf_path, dpi, pages)
            else:
                raise Exception(f"PDF画像変換エラー: {e}。popplerがインストールされていないか、PyMuPDFが利用できません。")
    else:
        # pdf2imageが利用できない場合、PyMuPDFを使用
        if PYMUPDF_AVAILABLE:
            return pdf_to_images_with_pymupdf(pdf_path, dpi, pages)
        else:
            raise ImportError("pdf2imageまたはPyMuPDFが必要です。pip install pdf2image PyMuPDF を実行してください。")


def pdf_to_images_from_bytes(pdf_bytes: bytes, dpi: int = 200, pages: Optional[List[int]] = None) -> List[np.ndarray]:
    """
    PDFバイトデータを画像に変換（画像レイヤーから）
    pdf2imageが失敗した場合、PyMuPDFをフォールバックとして使用
    
    Args:
        pdf_bytes: PDFファイルのバイトデータ
        dpi: 解像度（デフォルト200）
        pages: 処理するページ番号のリスト（Noneの場合は全ページ）
    
    Returns:
        画像のリスト（numpy配列）
    """
    # まずpdf2imageを試行
    if PDF2IMAGE_AVAILABLE:
        try:
            if pages is None:
                images = convert_from_bytes(pdf_bytes, dpi=dpi)
            else:
                images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=min(pages), last_page=max(pages))
                # 指定されたページのみを抽出
                if len(pages) < len(images):
                    images = [images[i - min(pages)] for i in pages if i - min(pages) < len(images)]
            
            # PIL Imageをnumpy配列に変換
            result = []
            for img in images:
                img_array = np.array(img)
                # RGBからBGRに変換（OpenCV用）
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                result.append(img_array)
            
            return result
        except Exception as e:
            # pdf2imageが失敗した場合（popplerがない場合など）、PyMuPDFをフォールバックとして使用
            if PYMUPDF_AVAILABLE:
                return pdf_to_images_with_pymupdf_bytes(pdf_bytes, dpi, pages)
            else:
                raise Exception(f"PDF画像変換エラー: {e}。popplerがインストールされていないか、PyMuPDFが利用できません。")
    else:
        # pdf2imageが利用できない場合、PyMuPDFを使用
        if PYMUPDF_AVAILABLE:
            return pdf_to_images_with_pymupdf_bytes(pdf_bytes, dpi, pages)
        else:
            raise ImportError("pdf2imageまたはPyMuPDFが必要です。pip install pdf2image PyMuPDF を実行してください。")


def pdf_to_images_with_pymupdf(pdf_path: str, dpi: int = 200, pages: Optional[List[int]] = None) -> List[np.ndarray]:
    """
    PDFファイルを画像に変換（PyMuPDFを使用、poppler不要）
    
    Args:
        pdf_path: PDFファイルのパス
        dpi: 解像度（デフォルト200、PyMuPDFでは拡大率として使用）
        pages: 処理するページ番号のリスト（Noneの場合は全ページ）
    
    Returns:
        画像のリスト（numpy配列）
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is not installed. Install it with: pip install PyMuPDF")
    
    try:
        import fitz
        doc = fitz.open(pdf_path)
        
        # 拡大率を計算（dpiを基準に、デフォルト72dpiから拡大）
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        target_pages = pages if pages is not None else list(range(len(doc)))
        
        result = []
        for page_num in target_pages:
            if page_num >= len(doc):
                continue
            
            page = doc[page_num]
            # ページを画像に変換
            pix = page.get_pixmap(matrix=mat)
            
            # PIL Imageに変換
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # numpy配列に変換
            img_array = np.array(img)
            # RGBからBGRに変換（OpenCV用）
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            result.append(img_array)
        
        doc.close()
        return result
    except Exception as e:
        raise Exception(f"PyMuPDF PDF画像変換エラー: {e}")


def pdf_to_images_with_pymupdf_bytes(pdf_bytes: bytes, dpi: int = 200, pages: Optional[List[int]] = None) -> List[np.ndarray]:
    """
    PDFバイトデータを画像に変換（PyMuPDFを使用、poppler不要）
    
    Args:
        pdf_bytes: PDFファイルのバイトデータ
        dpi: 解像度（デフォルト200、PyMuPDFでは拡大率として使用）
        pages: 処理するページ番号のリスト（Noneの場合は全ページ）
    
    Returns:
        画像のリスト（numpy配列）
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is not installed. Install it with: pip install PyMuPDF")
    
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # 拡大率を計算（dpiを基準に、デフォルト72dpiから拡大）
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        target_pages = pages if pages is not None else list(range(len(doc)))
        
        result = []
        for page_num in target_pages:
            if page_num >= len(doc):
                continue
            
            page = doc[page_num]
            # ページを画像に変換
            pix = page.get_pixmap(matrix=mat)
            
            # PIL Imageに変換
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # numpy配列に変換
            img_array = np.array(img)
            # RGBからBGRに変換（OpenCV用）
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            result.append(img_array)
        
        doc.close()
        return result
    except Exception as e:
        raise Exception(f"PyMuPDF PDF画像変換エラー: {e}")


def extract_text_from_pdf(pdf_path: str, pages: Optional[List[int]] = None) -> List[dict]:
    """
    PDFからテキストを抽出（テキストレイヤーから）
    
    Args:
        pdf_path: PDFファイルのパス
        pages: 処理するページ番号のリスト（Noneの場合は全ページ）
    
    Returns:
        各ページのテキスト情報のリスト
        [{'page': int, 'text': str, 'bboxes': List[dict]}, ...]
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is not installed. Install it with: pip install PyMuPDF")
    
    try:
        doc = fitz.open(pdf_path)
        results = []
        
        target_pages = pages if pages is not None else list(range(len(doc)))
        
        for page_num in target_pages:
            if page_num >= len(doc):
                continue
            
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            # テキストブロックの情報を取得
            text_blocks = []
            full_text = ""
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                            full_text += span["text"]
                        block_text += "\n"
                        full_text += "\n"
                    
                    if block_text.strip():
                        bbox = block["bbox"]
                        text_blocks.append({
                            "text": block_text.strip(),
                            "bbox": bbox  # [x0, y0, x1, y1]
                        })
            
            results.append({
                "page": page_num,
                "text": full_text.strip(),
                "blocks": text_blocks
            })
        
        doc.close()
        return results
    except Exception as e:
        raise Exception(f"PDFテキスト抽出エラー: {e}")


def extract_text_from_pdf_bytes(pdf_bytes: bytes, pages: Optional[List[int]] = None) -> List[dict]:
    """
    PDFバイトデータからテキストを抽出（テキストレイヤーから）
    
    Args:
        pdf_bytes: PDFファイルのバイトデータ
        pages: 処理するページ番号のリスト（Noneの場合は全ページ）
    
    Returns:
        各ページのテキスト情報のリスト
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is not installed. Install it with: pip install PyMuPDF")
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        results = []
        
        target_pages = pages if pages is not None else list(range(len(doc)))
        
        for page_num in target_pages:
            if page_num >= len(doc):
                continue
            
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            # テキストブロックの情報を取得
            text_blocks = []
            full_text = ""
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                            full_text += span["text"]
                        block_text += "\n"
                        full_text += "\n"
                    
                    if block_text.strip():
                        bbox = block["bbox"]
                        text_blocks.append({
                            "text": block_text.strip(),
                            "bbox": bbox  # [x0, y0, x1, y1]
                        })
            
            results.append({
                "page": page_num,
                "text": full_text.strip(),
                "blocks": text_blocks
            })
        
        doc.close()
        return results
    except Exception as e:
        raise Exception(f"PDFテキスト抽出エラー: {e}")


def preprocess_image_for_ocr(image: np.ndarray, coords: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    OCR精度向上のための画像前処理
    
    Args:
        image: 入力画像（BGR形式）
        coords: 切り出し座標 (x1, y1, x2, y2)、Noneの場合は画像全体
    
    Returns:
        前処理済み画像（グレースケール、二値化済み）
    """
    # 座標指定がある場合は領域を切り出し
    if coords is not None:
        x1, y1, x2, y2 = coords
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
    else:
        roi = image
    
    # グレースケール変換
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # 画像を拡大（OCR精度向上のため）
    scale_factor = 3
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # ノイズ除去
    gray = cv2.medianBlur(gray, 3)
    
    # 二値化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    画像ファイルを読み込む
    
    Args:
        file_path: 画像ファイルのパス
    
    Returns:
        画像データ（numpy配列）、読み込み失敗時はNone
    """
    try:
        image = cv2.imread(file_path)
        return image
    except Exception as e:
        print(f"画像読み込みエラー: {e}")
        return None


def image_to_bytes(image: np.ndarray, format: str = 'PNG') -> bytes:
    """
    画像をバイトデータに変換
    
    Args:
        image: 画像データ（numpy配列）
        format: 画像フォーマット（'PNG', 'JPEG'など）
    
    Returns:
        バイトデータ
    """
    if len(image.shape) == 3:
        # BGRからRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    pil_image = Image.fromarray(image_rgb)
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format=format)
    return img_bytes.getvalue()


def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """
    バイトデータを画像に変換
    
    Args:
        image_bytes: 画像のバイトデータ
    
    Returns:
        画像データ（numpy配列、BGR形式）
    """
    pil_image = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(pil_image)
    
    # RGBからBGRに変換（OpenCV用）
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array

