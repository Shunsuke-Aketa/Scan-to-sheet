"""
OCR抽出機能
PDF対応、日本語+英語OCR、ドラッグ選択対応
"""
import sys
from pathlib import Path

# プロジェクトルートをsys.pathに追加
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

import pytesseract
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import re
import platform
import os
from PIL import Image

# EasyOCRのインポート（オプション）
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

from src.utils import (
    pdf_to_images_from_path,
    pdf_to_images_from_bytes,
    extract_text_from_pdf,
    extract_text_from_pdf_bytes,
    preprocess_image_for_ocr,
    load_image,
    get_tesseract_path,
    get_tessdata_path,
    check_japanese_data,
    PYMUPDF_AVAILABLE
)


# Tesseractのパス設定
tesseract_path = get_tesseract_path()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # TESSDATA_PREFIX環境変数を設定
    tessdata_path = get_tessdata_path(tesseract_path)
    if tessdata_path:
        os.environ['TESSDATA_PREFIX'] = tessdata_path


def normalize_numbers(text: str) -> str:
    """
    OCR結果から丸数字を通常の数字に置換
    
    Args:
        text: OCR結果のテキスト
    
    Returns:
        丸数字が通常の数字に置換されたテキスト
    """
    if not text:
        return text
    
    # 丸数字のマッピング（①-⑳）
    circled_numbers_1_20 = {
        '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
        '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10',
        '⑪': '11', '⑫': '12', '⑬': '13', '⑭': '14', '⑮': '15',
        '⑯': '16', '⑰': '17', '⑱': '18', '⑲': '19', '⑳': '20'
    }
    
    # 丸数字のマッピング（㊀-㊉）
    circled_numbers_1_10 = {
        '㊀': '1', '㊁': '2', '㊂': '3', '㊃': '4', '㊄': '5',
        '㊅': '6', '㊆': '7', '㊇': '8', '㊈': '9', '㊉': '10'
    }
    
    # 置換を実行
    result = text
    for circled, normal in circled_numbers_1_20.items():
        result = result.replace(circled, normal)
    for circled, normal in circled_numbers_1_10.items():
        result = result.replace(circled, normal)
    
    return result


class OCRExtractor:
    """
    OCR抽出クラス
    画像・PDFからテキストを抽出する機能を提供
    TesseractとEasyOCRの両方に対応
    """
    
    def __init__(self, lang: str = 'eng+jpn', ocr_engine: str = 'tesseract'):
        """
        初期化
        
        Args:
            lang: OCR言語設定（デフォルト: 'eng+jpn'、英語を優先）
            ocr_engine: 使用するOCRエンジン ('tesseract' または 'easyocr')
        """
        # jpn+engが指定された場合はeng+jpnに変換（英語を優先するため）
        if lang == 'jpn+eng':
            lang = 'eng+jpn'
        self.lang = lang
        self.ocr_engine = ocr_engine
        
        # EasyOCRの初期化（使用する場合）
        self.easyocr_reader = None
        if ocr_engine == 'easyocr':
            if not EASYOCR_AVAILABLE:
                print("警告: EasyOCRがインストールされていません。Tesseractを使用します。")
                print("EasyOCRをインストールするには: pip install easyocr")
                self.ocr_engine = 'tesseract'
            else:
                # 日本語+英語の言語コードを設定
                lang_list = []
                if 'jpn' in lang or 'jpn' in lang.lower():
                    lang_list.append('ja')
                if 'eng' in lang or 'eng' in lang.lower():
                    lang_list.append('en')
                if not lang_list:
                    lang_list = ['en']  # デフォルトは英語
                
                try:
                    print(f"EasyOCRを初期化中... (言語: {lang_list})")
                    print("初回使用時はモデルのダウンロードに時間がかかります（約500MB）。")
                    self.easyocr_reader = easyocr.Reader(lang_list, gpu=False)
                    print("EasyOCRの初期化が完了しました。")
                except Exception as e:
                    print(f"EasyOCR初期化エラー: {e}")
                    print("Tesseractにフォールバックします")
                    self.ocr_engine = 'tesseract'
                    self.easyocr_reader = None
        
        # Tesseractの言語チェック（Tesseractを使用する場合）
        if self.ocr_engine == 'tesseract':
            self._check_tesseract_lang()
    
    def _check_tesseract_lang(self):
        """Tesseractで利用可能な言語を確認"""
        try:
            # まず日本語データファイルの存在を確認
            tessdata_path = get_tessdata_path()
            has_jpn_data = check_japanese_data(tessdata_path)
            
            if not has_jpn_data:
                tesseract_path = get_tesseract_path()
                if tesseract_path:
                    tessdata_dir = os.path.dirname(tesseract_path)
                    jpn_data_path = os.path.join(tessdata_dir, 'tessdata', 'jpn.traineddata')
                else:
                    jpn_data_path = "tessdata/jpn.traineddata"
                
                print("=" * 60)
                print("警告: 日本語データ（jpn.traineddata）が見つかりません。")
                print("=" * 60)
                print(f"日本語データファイルのパス: {jpn_data_path}")
                print("\n【解決方法】")
                print("1. 以下のURLから jpn.traineddata をダウンロードしてください:")
                print("   https://github.com/tesseract-ocr/tessdata/raw/main/jpn.traineddata")
                print(f"\n2. ダウンロードしたファイルを以下の場所に配置してください:")
                if tesseract_path:
                    print(f"   {os.path.join(os.path.dirname(tesseract_path), 'tessdata', 'jpn.traineddata')}")
                else:
                    print("   Tesseractのインストールディレクトリ\\tessdata\\jpn.traineddata")
                print("\n3. アプリを再起動してください。")
                print("=" * 60)
                
                # 言語リストも確認
                try:
                    available_langs = pytesseract.get_languages()
                    if 'jpn' not in available_langs:
                        print("警告: 日本語データ（jpn.traineddata）が見つかりません。")
                        print("英語のみで認識します。")
                        self.lang = 'eng'
                except Exception:
                    self.lang = 'eng'
            else:
                # 日本語データが存在する場合でも、pytesseractが認識できるか確認
                try:
                    available_langs = pytesseract.get_languages()
                    if 'jpn' not in available_langs:
                        print("警告: 日本語データファイルは存在しますが、Tesseractが認識できません。")
                        print("TESSDATA_PREFIX環境変数を確認してください。")
                        self.lang = 'eng'
                except Exception:
                    self.lang = 'eng'
        except Exception as e:
            print(f"言語確認エラー: {e}")
            self.lang = 'eng'
    
    def extract_text_from_region(
        self,
        image: np.ndarray,
        coords: Tuple[int, int, int, int],
        lang: Optional[str] = None
    ) -> str:
        """
        画像の指定領域からテキストを抽出
        
        Args:
            image: 入力画像（BGR形式）
            coords: 抽出領域の座標 (x1, y1, x2, y2)
            lang: OCR言語設定（Noneの場合は初期化時の設定を使用）
        
        Returns:
            抽出されたテキスト
        """
        try:
            lang = lang or self.lang
            
            # 画像の前処理
            processed = preprocess_image_for_ocr(image, coords)
            
            if processed is None:
                return ""
            
            # OCRエンジンに応じて処理
            if self.ocr_engine == 'easyocr' and self.easyocr_reader is not None:
                # EasyOCRを使用
                # EasyOCRはBGR形式を期待するので、RGBの場合はBGRに変換
                if len(processed.shape) == 3:
                    # 既にBGR形式の場合はそのまま使用
                    # preprocess_image_for_ocrが返す画像は通常BGR形式
                    processed_bgr = processed
                else:
                    processed_bgr = processed
                
                # EasyOCRでテキスト抽出
                results = self.easyocr_reader.readtext(processed_bgr)
                # EasyOCRの結果は [(bbox, text, confidence), ...] の形式
                # 信頼度が低い結果は除外（閾値: 0.3）
                text_parts = []
                for result in results:
                    if len(result) >= 2:
                        confidence = result[2] if len(result) >= 3 else 1.0
                        if confidence >= 0.3:  # 信頼度が30%以上の場合のみ使用
                            text_parts.append(result[1])
                
                text = ' '.join(text_parts)
            else:
                # Tesseractを使用（既存の処理）
                custom_config = f'--oem 3 --psm 6 -l {lang}'
                text = pytesseract.image_to_string(processed, config=custom_config)
            
            # テキストをクリーニング
            text = text.strip()
            
            # 丸数字を通常の数字に置換
            text = normalize_numbers(text)
            
            return text
        except Exception as e:
            print(f"テキスト抽出エラー: {e}")
            # エラー時はTesseractにフォールバック（EasyOCR使用中の場合）
            if self.ocr_engine == 'easyocr':
                try:
                    print("EasyOCRエラー、Tesseractにフォールバックします")
                    processed = preprocess_image_for_ocr(image, coords)
                    if processed is not None:
                        custom_config = f'--oem 3 --psm 6 -l {lang or self.lang}'
                        text = pytesseract.image_to_string(processed, config=custom_config)
                        text = normalize_numbers(text.strip())
                        return text
                except:
                    pass
            return ""
    
    def extract_text_from_regions(
        self,
        image: np.ndarray,
        regions: List[Dict[str, Union[Tuple[int, int, int, int], str]]],
        lang: Optional[str] = None
    ) -> Dict[str, str]:
        """
        複数の領域からテキストを抽出
        
        Args:
            image: 入力画像（BGR形式）
            regions: 領域のリスト [{'coords': (x1, y1, x2, y2), 'name': '範囲1'}, ...]
            lang: OCR言語設定
        
        Returns:
            領域名をキーとした抽出テキストの辞書
        """
        results = {}
        for region in regions:
            coords = region.get('coords')
            name = region.get('name', f"{len(results) + 1}")
            
            if coords:
                text = self.extract_text_from_region(image, coords, lang)
                results[name] = text
        
        return results
    
    def process_image(
        self,
        image_path: str,
        regions: List[Dict[str, Union[Tuple[int, int, int, int], str]]]
    ) -> Dict[str, any]:
        """
        画像ファイルを処理してテキストを抽出
        
        Args:
            image_path: 画像ファイルのパス
            regions: 抽出領域のリスト
        
        Returns:
            抽出結果の辞書
        """
        try:
            # PIL Imageを使用して画像を読み込む（cv2.imreadより確実）
            pil_image = Image.open(image_path)
            # PIL Imageをnumpy配列に変換（BGR形式）
            img_array = np.array(pil_image)
            # RGBからBGRに変換（OpenCV用）
            if len(img_array.shape) == 3:
                image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                image = img_array
        except Exception as e:
            print(f"画像読み込みエラー: {e}")
            return {
                'filename': Path(image_path).name,
                'error': f'画像の読み込みに失敗しました: {str(e)}'
            }
        
        extracted_texts = self.extract_text_from_regions(image, regions)
        
        result = {
            'filename': Path(image_path).name,
            'filepath': image_path,
            **extracted_texts
        }
        
        return result
    
    def process_pdf_page(
        self,
        pdf_path: str,
        page_num: int,
        regions: List[Dict[str, Union[Tuple[int, int, int, int], str]]],
        use_text_layer: bool = True
    ) -> Dict[str, any]:
        """
        PDFの1ページを処理
        
        Args:
            pdf_path: PDFファイルのパス
            page_num: ページ番号（0始まり）
            regions: 抽出領域のリスト
            use_text_layer: テキストレイヤーを優先するか（True: テキストレイヤー優先、False: 画像レイヤー優先）
        
        Returns:
            抽出結果の辞書
        """
        result = {
            'filename': Path(pdf_path).name,
            'filepath': pdf_path,
            'page': page_num + 1,  # 1始まりで表示
            'page_index': page_num  # 0始まりのインデックス
        }
        
        # テキストレイヤーから抽出を試行
        text_layer_results = {}
        if use_text_layer:
            try:
                text_data = extract_text_from_pdf(pdf_path, pages=[page_num])
                if text_data and len(text_data) > 0:
                    page_text_data = text_data[0]
                    
                    # 各領域に対応するテキストを抽出
                    for region in regions:
                        coords = region.get('coords')
                        name = region.get('name', f"{len(text_layer_results) + 1}")
                        
                        if coords:
                            # 座標範囲内のテキストブロックを検索
                            x1, y1, x2, y2 = coords
                            matched_texts = []
                            
                            for block in page_text_data.get('blocks', []):
                                bbox = block.get('bbox', [])
                                if len(bbox) == 4:
                                    bx1, by1, bx2, by2 = bbox
                                    # 領域と重なるテキストブロックを検索
                                    if not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2):
                                        matched_texts.append(block.get('text', ''))
                            
                            text = '\n'.join(matched_texts).strip()
                            # 丸数字を通常の数字に置換
                            text_layer_results[name] = normalize_numbers(text)
            except Exception as e:
                print(f"テキストレイヤー抽出エラー: {e}")
        
        # 画像レイヤーからOCR抽出を試行
        image_layer_results = {}
        try:
            images = pdf_to_images_from_path(pdf_path, pages=[page_num])
            if images and len(images) > 0:
                image = images[0]
                image_layer_results = self.extract_text_from_regions(image, regions)
        except Exception as e:
            print(f"画像レイヤー抽出エラー: {e}")
        
        # 精度の高い方を選択（テキストが長い方を優先、ただし空でない方を優先）
        final_results = {}
        for region in regions:
            name = region.get('name', f"{len(final_results) + 1}")
            
            text_layer_text = text_layer_results.get(name, '').strip()
            image_layer_text = image_layer_results.get(name, '').strip()
            
            # 精度判定：テキストが存在し、長い方を選択
            if text_layer_text and image_layer_text:
                # 両方ある場合は、より長い方を選択（より多くの情報を含む）
                if len(text_layer_text) >= len(image_layer_text):
                    final_results[name] = text_layer_text
                else:
                    final_results[name] = image_layer_text
            elif text_layer_text:
                final_results[name] = text_layer_text
            elif image_layer_text:
                final_results[name] = image_layer_text
            else:
                final_results[name] = ""
        
        result.update(final_results)
        return result
    
    def process_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        regions: List[Dict[str, Union[Tuple[int, int, int, int], str]]] = None,
        use_text_layer: bool = True
    ) -> List[Dict[str, any]]:
        """
        PDFファイルを処理（全ページまたは指定ページ）
        
        Args:
            pdf_path: PDFファイルのパス
            pages: 処理するページ番号のリスト（Noneの場合は全ページ、0始まり）
            regions: 抽出領域のリスト
            use_text_layer: テキストレイヤーを優先するか
        
        Returns:
            各ページの抽出結果のリスト
        """
        if regions is None:
            regions = []
        
        try:
            # PDFのページ数を取得
            total_pages = None
            if PYMUPDF_AVAILABLE:
                try:
                    import fitz
                    doc = fitz.open(pdf_path)
                    total_pages = len(doc)
                    doc.close()
                except Exception:
                    pass
            
            if pages is None:
                # 全ページを処理
                if total_pages is not None:
                    pages = list(range(total_pages))
                else:
                    # ページ数が不明な場合は最初の10ページまで処理
                    pages = list(range(10))
            
            results = []
            for page_num in pages:
                try:
                    page_result = self.process_pdf_page(
                        pdf_path, page_num, regions, use_text_layer
                    )
                    results.append(page_result)
                except Exception as e:
                    print(f"ページ {page_num + 1} の処理エラー: {e}")
                    results.append({
                        'filename': Path(pdf_path).name,
                        'page': page_num + 1,
                        'error': str(e)
                    })
            
            return results
        except Exception as e:
            print(f"PDF処理エラー: {e}")
            return [{
                'filename': Path(pdf_path).name,
                'error': str(e)
            }]
    
    def process_file(
        self,
        file_path: str,
        regions: List[Dict[str, Union[Tuple[int, int, int, int], str]]],
        pages: Optional[List[int]] = None
    ) -> Union[Dict[str, any], List[Dict[str, any]]]:
        """
        ファイルを処理（画像またはPDF）
        
        Args:
            file_path: ファイルのパス
            regions: 抽出領域のリスト
            pages: PDFの場合のページ番号リスト（Noneの場合は全ページ）
        
        Returns:
            抽出結果（画像の場合は辞書、PDFの場合はリスト）
        """
        file_path_obj = Path(file_path)
        ext = file_path_obj.suffix.lower()
        
        if ext == '.pdf':
            return self.process_pdf(file_path, pages, regions)
        else:
            # 画像ファイルとして処理
            return self.process_image(file_path, regions)



