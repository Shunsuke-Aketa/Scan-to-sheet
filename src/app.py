"""
Streamlit Webアプリケーション
画像・PDFからOCRでテキストを抽出するツール
"""
import sys
from pathlib import Path

# プロジェクトルートをsys.pathに追加
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from typing import List, Dict, Tuple, Optional
import json
import io
import base64
import threading
import time

from PIL import Image
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    CV2_AVAILABLE = False
    cv2 = None
    st.error(f"⚠️ OpenCVのインポートに失敗しました: {e}\n\n画像処理機能が制限される可能性があります。")

from src.extractors import OCRExtractor, EASYOCR_AVAILABLE
from src.utils import (
    pdf_to_images_from_path,
    load_image,
    image_to_bytes,
    bytes_to_image,
    get_tesseract_path,
    get_tessdata_path,
    check_japanese_data,
    PYMUPDF_AVAILABLE
)

# ページ設定
st.set_page_config(
    page_title="Scan To Sheet - OCR抽出ツール",
    page_icon="📄",
    layout="wide"
)

# セッション状態の初期化
# EasyOCRが利用可能な場合はデフォルトでEasyOCRを使用（Streamlit Cloud対応）
default_engine = 'easyocr' if EASYOCR_AVAILABLE else 'tesseract'
if 'extractor' not in st.session_state:
    st.session_state.extractor = OCRExtractor(lang='eng+jpn', ocr_engine=default_engine)

if 'ocr_engine' not in st.session_state:
    st.session_state.ocr_engine = default_engine

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'selected_regions' not in st.session_state:
    st.session_state.selected_regions = []

if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []

if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0

if 'current_image' not in st.session_state:
    st.session_state.current_image = None

if 'current_file_type' not in st.session_state:
    st.session_state.current_file_type = None

if 'selected_files_for_processing' not in st.session_state:
    st.session_state.selected_files_for_processing = []


def convert_image_for_display(image) -> Image.Image:
    """
    OpenCV画像またはPIL ImageをPIL Imageに変換（表示用）
    
    Args:
        image: OpenCV画像（np.ndarray）またはPIL Image
    
    Returns:
        PIL Image
    """
    # 既にPIL Imageの場合はそのまま返す
    if isinstance(image, Image.Image):
        return image
    
    # numpy配列の場合はPIL Imageに変換
    if isinstance(image, np.ndarray):
        if not CV2_AVAILABLE or cv2 is None:
            # OpenCVが利用できない場合は、そのままPIL Imageに変換
            return Image.fromarray(image)
        
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        return Image.fromarray(image_rgb)
    
    # その他の型の場合はエラー
    raise TypeError(f"Unsupported image type: {type(image)}")


def image_to_base64(image: Image.Image) -> str:
    """
    PIL Imageをbase64エンコードしたdata URIに変換
    
    Args:
        image: PIL Image
    
    Returns:
        base64エンコードされたdata URI文字列
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def visualize_regions_on_image(image: Image.Image, regions: List[Dict]) -> Image.Image:
    """
    画像上に範囲を可視化（矩形を描画）
    
    Args:
        image: 元の画像（PIL Image）
        regions: 範囲のリスト
    
    Returns:
        範囲が描画された画像（PIL Image）
    """
    # OpenCVが利用できない場合は、元の画像をそのまま返す
    if not CV2_AVAILABLE or cv2 is None:
        return image
    
    # PIL Imageをnumpy配列に変換
    img_array = np.array(image)
    
    # RGBからBGRに変換（OpenCV用）
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # 各範囲を描画
    colors = [
        (0, 0, 255),    # 赤
        (0, 255, 0),    # 緑
        (255, 0, 0),    # 青
        (0, 255, 255),  # 黄
        (255, 0, 255),  # マゼンタ
        (255, 255, 0),  # シアン
    ]
    
    for i, region in enumerate(regions):
        coords = region['coords']
        x1, y1, x2, y2 = coords
        
        # 色を選択（範囲数に応じて）
        color = colors[i % len(colors)]
        
        # 矩形を描画
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        
        # 範囲名を描画
        cv2.putText(img_bgr, region['name'], (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # BGRからRGBに変換してPIL Imageに戻す
    if len(img_bgr.shape) == 3:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr
    
    return Image.fromarray(img_rgb)


def open_opencv_coord_picker(image: Image.Image, image_key: str) -> Optional[List[Dict[str, Tuple[int, int]]]]:
    """
    OpenCVウィンドウを開いて画像上で複数の範囲（左上、右下の2点）を連続選択して座標を取得
    
    Args:
        image: 表示する画像（PIL Image）
        image_key: セッション状態で管理するキー
    
    Returns:
        座標のリスト [{'top_left': (x, y), 'bottom_right': (x, y)}, ...] または None
    
    Note:
        StreamlitはWebアプリのため、この関数はサーバー側で実行されます。
        ローカルで実行している場合（streamlit run）のみ、OpenCVウィンドウが表示されます。
        リモートサーバーで実行している場合は、ウィンドウが表示されない可能性があります。
        
        操作方法:
        - 左上の点をクリック
        - 右下の点をクリック
        - Enterキーで範囲を確定して次の範囲選択に進む
        - ESCキーで終了してすべての範囲を返す
    """
    try:
        # OpenCVが利用可能か確認
        if not CV2_AVAILABLE or cv2 is None:
            raise RuntimeError(
                "OpenCVが利用できません。\n"
                "この環境ではOpenCVウィンドウを使用できません。\n"
                "数値入力フィールドで座標を手動入力してください。"
            )
        if not hasattr(cv2, 'imshow'):
            raise RuntimeError("OpenCVが正しくインストールされていません。")
        
        # OpenCVのGUIサポートを確認（namedWindowが使用可能かテスト）
        try:
            test_window_name = '__opencv_test_window__'
            cv2.namedWindow(test_window_name, cv2.WINDOW_NORMAL)
            cv2.destroyWindow(test_window_name)
        except cv2.error as e:
            # GUIサポートがない場合
            error_msg = str(e)
            if "not implemented" in error_msg.lower() or "gtk" in error_msg.lower() or "cocoa" in error_msg.lower():
                raise RuntimeError(
                    "OpenCVのGUIサポートが利用できません。\n"
                    "この環境ではOpenCVウィンドウを使用できません。\n"
                    "数値入力フィールドで座標を手動入力してください。\n\n"
                    "解決方法:\n"
                    "1. `opencv-python-headless`がインストールされている場合は、アンインストールしてください\n"
                    "2. `pip uninstall opencv-python-headless`\n"
                    "3. `pip install opencv-python` で再インストールしてください"
                )
            else:
                raise
        
        # PIL ImageをOpenCV形式に変換
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            # RGBからBGRに変換
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # 画像サイズを取得
        img_height, img_width = img_bgr.shape[:2]
        
        # 画面サイズの80%を最大サイズとして使用（デフォルト値）
        max_width = 1536  # 1920 * 0.8
        max_height = 864  # 1080 * 0.8
        
        # アスペクト比を保持しながらリサイズ
        scale = min(max_width / img_width, max_height / img_height, 1.0)
        if scale < 1.0:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            display_img = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # リサイズ後の座標を元の画像座標に変換するためのスケール
            scale_factor = 1.0 / scale
        else:
            display_img = img_bgr.copy()
            scale_factor = 1.0
        
        # 表示用画像のコピーを作成
        base_img = display_img.copy()
        
        # 選択済みの範囲を保存するリスト
        confirmed_regions = []
        
        # クリック座標を保存する変数
        clicked_points = {
            'top_left': None,
            'bottom_right': None
        }
        click_count = 0
        
        # 色のリスト（各範囲に異なる色を割り当て）
        colors = [
            (0, 0, 255),    # 赤
            (0, 255, 0),    # 緑
            (255, 0, 0),    # 青
            (0, 255, 255),  # 黄
            (255, 0, 255),  # マゼンタ
            (255, 255, 0),  # シアン
        ]
        
        def draw_all_regions():
            """選択済みの範囲と現在選択中の範囲を描画"""
            display_img_ref = base_img.copy()
            circle_size = max(5, int(10 * scale)) if scale < 1.0 else 10
            line_thickness = max(1, int(2 * scale)) if scale < 1.0 else 2
            
            # 選択済みの範囲を描画
            for i, region in enumerate(confirmed_regions):
                color = colors[i % len(colors)]
                top_left = region['top_left']
                bottom_right = region['bottom_right']
                
                # 元の座標を表示用座標に変換
                top_left_display = (
                    int(top_left[0] * scale),
                    int(top_left[1] * scale)
                )
                bottom_right_display = (
                    int(bottom_right[0] * scale),
                    int(bottom_right[1] * scale)
                )
                
                # 矩形を描画
                cv2.rectangle(display_img_ref, top_left_display, bottom_right_display, color, line_thickness)
                # 点を描画
                cv2.circle(display_img_ref, top_left_display, circle_size, color, -1)
                cv2.circle(display_img_ref, bottom_right_display, circle_size, color, -1)
            
            # 現在選択中の範囲を描画
            if clicked_points['top_left'] is not None:
                top_left_display = (
                    int(clicked_points['top_left'][0] * scale),
                    int(clicked_points['top_left'][1] * scale)
                )
                cv2.circle(display_img_ref, top_left_display, circle_size, (0, 0, 255), -1)  # 赤
                
                if clicked_points['bottom_right'] is not None:
                    bottom_right_display = (
                        int(clicked_points['bottom_right'][0] * scale),
                        int(clicked_points['bottom_right'][1] * scale)
                    )
                    cv2.circle(display_img_ref, bottom_right_display, circle_size, (0, 255, 0), -1)  # 緑
                    cv2.rectangle(display_img_ref, top_left_display, bottom_right_display, (255, 0, 255), line_thickness)  # マゼンタ
            
            return display_img_ref
        
        # マウスコールバック関数
        def mouse_callback(event, x, y, flags, param):
            nonlocal click_count, clicked_points
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # 表示用画像の座標を元の画像座標に変換
                orig_x = int(x * scale_factor) if scale_factor != 1.0 else x
                orig_y = int(y * scale_factor) if scale_factor != 1.0 else y
                
                if click_count == 0:
                    # 1回目のクリック: 左上の点
                    clicked_points['top_left'] = (orig_x, orig_y)
                    clicked_points['bottom_right'] = None
                    click_count = 1
                    print(f"[OpenCV] 左上の点を選択: ({orig_x}, {orig_y})")
                elif click_count == 1:
                    # 2回目のクリック: 右下の点
                    clicked_points['bottom_right'] = (orig_x, orig_y)
                    click_count = 2
                    print(f"[OpenCV] 右下の点を選択: ({orig_x}, {orig_y})")
                
                # 画像を再描画
                display_img_ref = draw_all_regions()
                cv2.imshow(window_name, display_img_ref)
        
        # ウィンドウを作成
        window_name = 'Coordinate Picker'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # ウィンドウサイズを画像サイズに設定（アスペクト比を保持）
        cv2.resizeWindow(window_name, base_img.shape[1], base_img.shape[0])
        cv2.imshow(window_name, base_img)
        
        # ウィンドウが閉じられるまで待機
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESCキーで終了
                print(f"[OpenCV] 終了しました。{len(confirmed_regions)} 個の範囲を取得しました。")
                cv2.destroyAllWindows()
                return confirmed_regions if confirmed_regions else None
            elif key == 13 or key == 10:  # Enterキーで範囲を確定
                if click_count >= 2 and clicked_points['top_left'] and clicked_points['bottom_right']:
                    # 範囲を確定してリストに追加
                    confirmed_regions.append({
                        'top_left': clicked_points['top_left'],
                        'bottom_right': clicked_points['bottom_right']
                    })
                    print(f"[OpenCV] 範囲 {len(confirmed_regions)} を確定: {clicked_points}")
                    # 次の範囲選択のためにリセット
                    clicked_points = {
                        'top_left': None,
                        'bottom_right': None
                    }
                    click_count = 0
                    # 画像を再描画
                    display_img_ref = draw_all_regions()
                    cv2.imshow(window_name, display_img_ref)
            elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                # ウィンドウが閉じられた
                print(f"[OpenCV] ウィンドウが閉じられました。{len(confirmed_regions)} 個の範囲を取得しました。")
                return confirmed_regions if confirmed_regions else None
        
    except RuntimeError as e:
        # OpenCVが利用できない場合
        print(f"[OpenCV] ランタイムエラー: {e}")
        raise
    except Exception as e:
        # その他のエラー
        print(f"[OpenCV] エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        raise


def draw_point_on_image(image: Image.Image, x: int, y: int, color: Tuple[int, int, int] = (255, 0, 0), size: int = 5) -> Image.Image:
    """
    画像上に点を描画
    
    Args:
        image: 元の画像（PIL Image）
        x: X座標
        y: Y座標
        color: 点の色（RGB）
        size: 点のサイズ（半径）
    
    Returns:
        点が描画された画像（PIL Image）
    """
    # OpenCVが利用できない場合は、元の画像をそのまま返す
    if not CV2_AVAILABLE or cv2 is None:
        return image
    
    img_array = np.array(image)
    
    # RGBからBGRに変換（OpenCV用）
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # 点を描画
    cv2.circle(img_bgr, (x, y), size, color[::-1], -1)  # color[::-1]でRGB→BGRに変換
    
    # BGRからRGBに変換してPIL Imageに戻す
    if len(img_bgr.shape) == 3:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr
    
    return Image.fromarray(img_rgb)


def create_image_with_coord_display(image: Image.Image, image_key: str, original_width: int = None, original_height: int = None) -> str:
    """
    画像を表示し、カーソル位置の座標を表示し、クリック座標を取得するHTMLコンポーネントを作成
    
    Args:
        image: 表示する画像（PIL Image、リサイズ済みの可能性あり）
        image_key: セッション状態で管理するキー
        original_width: 元の画像の幅（リサイズ前、Noneの場合はimage.widthを使用）
        original_height: 元の画像の高さ（リサイズ前、Noneの場合はimage.heightを使用）
    
    Returns:
        HTML文字列
    """
    # 画像をbase64エンコード
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # ユニークなIDを生成（特殊文字を置換）
    unique_id = image_key.replace(" ", "_").replace(".", "_").replace("/", "_").replace("\\", "_")
    
    # 元の画像サイズを取得（座標変換用）
    # セッション状態から取得を試みる
    if original_width is None or original_height is None:
        original_size_key = f'original_image_size_{image_key}'
        if original_size_key in st.session_state:
            original_width, original_height = st.session_state[original_size_key]
        else:
            # セッション状態にない場合は、表示画像のサイズを使用
            original_width = image.width
            original_height = image.height
    
    # 表示画像のサイズ
    display_width = image.width
    display_height = image.height
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                margin: 0;
                padding: 10px;
                font-family: Arial, sans-serif;
            }}
            #container_{unique_id} {{
                position: relative;
                display: inline-block;
                width: 100%;
                max-width: 100%;
            }}
            #coord_image_{unique_id} {{
                max-width: 100%;
                height: auto;
                cursor: crosshair;
                display: block;
                user-select: none;
            }}
            #coord_display_{unique_id} {{
                position: absolute;
                background: rgba(0, 0, 0, 0.85);
                color: white;
                padding: 8px 12px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                pointer-events: none;
                display: none;
                z-index: 1000;
                white-space: nowrap;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            }}
        </style>
    </head>
    <body>
        <div id="container_{unique_id}">
            <img id="coord_image_{unique_id}" 
                 src="data:image/png;base64,{img_str}" 
                 alt="座標選択用画像" />
            <div id="coord_display_{unique_id}"></div>
        </div>
        <script>
            (function() {{
                'use strict';
                
                let img_{unique_id} = null;
                let display_{unique_id} = null;
                let container_{unique_id} = null;
                let originalWidth_{unique_id} = {original_width};
                let originalHeight_{unique_id} = {original_height};
                let displayWidth_{unique_id} = {display_width};
                let displayHeight_{unique_id} = {display_height};
                
                function initImage_{unique_id}() {{
                    img_{unique_id} = document.getElementById('coord_image_{unique_id}');
                    display_{unique_id} = document.getElementById('coord_display_{unique_id}');
                    container_{unique_id} = document.getElementById('container_{unique_id}');
                    
                    if (img_{unique_id} && display_{unique_id}) {{
                        // 画像の読み込み完了を待つ
                        if (img_{unique_id}.complete) {{
                            attachEventListeners_{unique_id}();
                        }} else {{
                            img_{unique_id}.addEventListener('load', attachEventListeners_{unique_id});
                        }}
                    }}
                }}
                
                function attachEventListeners_{unique_id}() {{
                    if (!img_{unique_id} || !display_{unique_id}) return;
                    
                    img_{unique_id}.addEventListener('mousemove', showCoordinates_{unique_id});
                    img_{unique_id}.addEventListener('mouseleave', hideCoordinates_{unique_id});
                    img_{unique_id}.addEventListener('click', handleImageClick_{unique_id});
                }}
                
                function showCoordinates_{unique_id}(event) {{
                    if (!img_{unique_id} || !display_{unique_id}) return;
                    
                    const rect = img_{unique_id}.getBoundingClientRect();
                    // 表示画像のサイズと元の画像サイズの比率を計算
                    const scaleX = originalWidth_{unique_id} / displayWidth_{unique_id};
                    const scaleY = originalHeight_{unique_id} / displayHeight_{unique_id};
                    
                    // クリック位置を表示画像の座標に変換
                    const displayX = (event.clientX - rect.left) * (displayWidth_{unique_id} / rect.width);
                    const displayY = (event.clientY - rect.top) * (displayHeight_{unique_id} / rect.height);
                    
                    // 元の画像の座標に変換
                    const x = Math.round(displayX * scaleX);
                    const y = Math.round(displayY * scaleY);
                    
                    // 座標を表示範囲内に制限
                    const clampedX = Math.max(0, Math.min(x, originalWidth_{unique_id} - 1));
                    const clampedY = Math.max(0, Math.min(y, originalHeight_{unique_id} - 1));
                    
                    display_{unique_id}.textContent = `座標: (${{clampedX}}, ${{clampedY}})`;
                    display_{unique_id}.style.display = 'block';
                    
                    const offsetX = event.clientX - rect.left + 15;
                    const offsetY = event.clientY - rect.top - 35;
                    
                    display_{unique_id}.style.left = offsetX + 'px';
                    display_{unique_id}.style.top = offsetY + 'px';
                }}
                
                function hideCoordinates_{unique_id}() {{
                    if (display_{unique_id}) {{
                        display_{unique_id}.style.display = 'none';
                    }}
                }}
                
                function handleImageClick_{unique_id}(event) {{
                    if (!img_{unique_id}) return;
                    
                    event.preventDefault();
                    event.stopPropagation();
                    
                    const rect = img_{unique_id}.getBoundingClientRect();
                    // 表示画像のサイズと元の画像サイズの比率を計算
                    const scaleX = originalWidth_{unique_id} / displayWidth_{unique_id};
                    const scaleY = originalHeight_{unique_id} / displayHeight_{unique_id};
                    
                    // クリック位置を表示画像の座標に変換
                    const displayX = (event.clientX - rect.left) * (displayWidth_{unique_id} / rect.width);
                    const displayY = (event.clientY - rect.top) * (displayHeight_{unique_id} / rect.height);
                    
                    // 元の画像の座標に変換
                    const x = Math.round(displayX * scaleX);
                    const y = Math.round(displayY * scaleY);
                    
                    // 座標を表示範囲内に制限
                    const clampedX = Math.max(0, Math.min(x, originalWidth_{unique_id} - 1));
                    const clampedY = Math.max(0, Math.min(y, originalHeight_{unique_id} - 1));
                    
                    console.log('[CLICK] クリック座標（元の画像）:', clampedX, clampedY);
                    console.log('[CLICK] 元の画像サイズ:', originalWidth_{unique_id}, originalHeight_{unique_id});
                    console.log('[CLICK] 表示画像サイズ:', displayWidth_{unique_id}, displayHeight_{unique_id});
                    console.log('[CLICK] 表示領域サイズ:', rect.width, rect.height);
                    console.log('[CLICK] スケール:', scaleX, scaleY);
                    
                    // URLパラメータを使用してStreamlitに座標を送信
                    const timestamp = Date.now();
                    const params = new URLSearchParams({{
                        'click_x': clampedX.toString(),
                        'click_y': clampedY.toString(),
                        'image_key': '{image_key}',
                        'timestamp': timestamp.toString()
                    }});
                    
                    console.log('[CLICK] URLパラメータ:', params.toString());
                    
                    // Streamlitの親ウィンドウにURLパラメータを送信
                    // 複数の方法を試行
                    let urlUpdated = false;
                    
                    // 方法1: window.parent.postMessageを使用（推奨）
                    try {{
                        if (window.parent && window.parent !== window) {{
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue',
                                value: {{
                                    click_x: clampedX,
                                    click_y: clampedY,
                                    image_key: '{image_key}',
                                    timestamp: timestamp
                                }}
                            }}, '*');
                            console.log('[CLICK] postMessageで送信しました');
                        }}
                    }} catch (e) {{
                        console.log('[CLICK] postMessageエラー:', e);
                    }}
                    
                    // 方法2: window.top.location.hrefを使用
                    if (!urlUpdated) {{
                        try {{
                            if (window.top && window.top !== window) {{
                                const currentUrl = window.top.location.href.split('?')[0];
                                const newUrl = currentUrl + '?' + params.toString();
                                console.log('[CLICK] window.top.location.hrefを更新:', newUrl);
                                window.top.location.href = newUrl;
                                urlUpdated = true;
                            }}
                        }} catch (e) {{
                            console.log('[CLICK] window.top.location.hrefエラー:', e);
                        }}
                    }}
                    
                    // 方法3: window.parent.location.hrefを使用
                    if (!urlUpdated) {{
                        try {{
                            if (window.parent && window.parent !== window) {{
                                const currentUrl = window.parent.location.href.split('?')[0];
                                const newUrl = currentUrl + '?' + params.toString();
                                console.log('[CLICK] window.parent.location.hrefを更新:', newUrl);
                                window.parent.location.href = newUrl;
                                urlUpdated = true;
                            }}
                        }} catch (e) {{
                            console.log('[CLICK] window.parent.location.hrefエラー:', e);
                        }}
                    }}
                    
                    // 方法4: 現在のウィンドウのURLを変更（フォールバック）
                    if (!urlUpdated) {{
                        try {{
                            const currentUrl = window.location.href.split('?')[0];
                            const newUrl = currentUrl + '?' + params.toString();
                            console.log('[CLICK] window.location.hrefを更新:', newUrl);
                            window.location.href = newUrl;
                        }} catch (e) {{
                            console.error('[CLICK] すべてのURL更新方法が失敗しました:', e);
                        }}
                    }}
                }}
                
                // ページ読み込み時に初期化
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', initImage_{unique_id});
                }} else {{
                    initImage_{unique_id}();
                }}
            }})();
        </script>
    </body>
    </html>
    """
    return html


def render_click_coord_input(image: Image.Image, image_key: str) -> List[Dict]:
    """
    画像上で2点（左上と右下）をクリックして範囲を選択するUIを表示
    カーソル位置の座標をリアルタイムで表示
    
    Args:
        image: 表示する画像（PIL Image）
        image_key: セッション状態で管理するキー
    
    Returns:
        矩形のリスト [{'coords': (x1, y1, x2, y2), 'name': '範囲1'}, ...]
    """
    print(f"[DEBUG] render_click_coord_input が呼び出されました: image_key={image_key}")
    
    # セッション状態で選択された範囲を管理
    if f'click_regions_{image_key}' not in st.session_state:
        st.session_state[f'click_regions_{image_key}'] = []
    
    # 現在編集中の範囲の2点を管理
    if f'current_points_{image_key}' not in st.session_state:
        st.session_state[f'current_points_{image_key}'] = {
            'top_left': None,
            'bottom_right': None
        }
    
    regions = st.session_state[f'click_regions_{image_key}']
    current_points = st.session_state[f'current_points_{image_key}']
    
    # クリック回数を追跡するセッション状態
    if f'click_count_{image_key}' not in st.session_state:
        st.session_state[f'click_count_{image_key}'] = 0
    
    # 処理済みクリックを追跡するセッション状態
    if f'processed_clicks_{image_key}' not in st.session_state:
        st.session_state[f'processed_clicks_{image_key}'] = set()
    
    # URLパラメータからクリック座標を読み取る
    query_params = st.query_params
    
    # デバッグ: URLパラメータの内容を表示
    if query_params:
        print(f"[DEBUG] query_params: {dict(query_params)}")
    
    # 処理済みクリックIDを追跡するキー
    processed_click_key = f'processed_click_{image_key}'
    
    if 'click_x' in query_params and 'click_y' in query_params and 'image_key' in query_params:
        click_image_key = query_params.get('image_key', '')
        print(f"[DEBUG] クリック座標を受信: image_key={click_image_key}, 現在のimage_key={image_key}")
        
        if click_image_key == image_key:
            try:
                click_x_str = query_params.get('click_x', '0')
                click_y_str = query_params.get('click_y', '0')
                timestamp = query_params.get('timestamp', '0')
                
                print(f"[DEBUG] 座標文字列: click_x={click_x_str}, click_y={click_y_str}, timestamp={timestamp}")
                
                click_x = int(click_x_str)
                click_y = int(click_y_str)
                
                print(f"[DEBUG] 座標整数: click_x={click_x}, click_y={click_y}")
                print(f"[DEBUG] 画像サイズ: width={image.width}, height={image.height}")
                
                # 座標が有効な範囲内かチェック
                if 0 <= click_x <= image.width and 0 <= click_y <= image.height:
                    # 処理済みかどうかをチェック（タイムスタンプを使用）
                    click_id = f"{click_x}_{click_y}_{timestamp}"
                    last_processed_id = st.session_state.get(processed_click_key, '')
                    
                    print(f"[DEBUG] クリックID: {click_id}, 前回処理済みID: {last_processed_id}")
                    
                    if click_id != last_processed_id:
                        # クリック回数を取得
                        click_count = st.session_state[f'click_count_{image_key}']
                        
                        print(f"[DEBUG] クリック回数: {click_count}")
                        
                        # 1回目のクリックは左上、2回目のクリックは右下
                        if click_count % 2 == 0:
                            # 左上の点を設定
                            current_points['top_left'] = (click_x, click_y)
                            st.session_state[f'click_count_{image_key}'] = click_count + 1
                            st.success(f"✅ 左上の点を選択しました: ({click_x}, {click_y})")
                            print(f"[DEBUG] 左上の点を設定: ({click_x}, {click_y})")
                        else:
                            # 右下の点を設定
                            current_points['bottom_right'] = (click_x, click_y)
                            st.session_state[f'click_count_{image_key}'] = click_count + 1
                            st.success(f"✅ 右下の点を選択しました: ({click_x}, {click_y})")
                            print(f"[DEBUG] 右下の点を設定: ({click_x}, {click_y})")
                        
                        # セッション状態を更新
                        st.session_state[f'current_points_{image_key}'] = current_points
                        st.session_state[processed_click_key] = click_id
                        
                        # 数値入力フィールドのセッション状態も更新
                        if current_points['top_left']:
                            st.session_state[f'top_left_x_{image_key}'] = current_points['top_left'][0]
                            st.session_state[f'top_left_y_{image_key}'] = current_points['top_left'][1]
                            print(f"[DEBUG] 数値入力フィールドを更新: top_left=({current_points['top_left'][0]}, {current_points['top_left'][1]})")
                        if current_points['bottom_right']:
                            st.session_state[f'bottom_right_x_{image_key}'] = current_points['bottom_right'][0]
                            st.session_state[f'bottom_right_y_{image_key}'] = current_points['bottom_right'][1]
                            print(f"[DEBUG] 数値入力フィールドを更新: bottom_right=({current_points['bottom_right'][0]}, {current_points['bottom_right'][1]})")
                        
                        # URLパラメータをクリアしてリロード
                        # 新しいクエリパラメータを作成（クリックパラメータを除く）
                        new_params = dict(query_params)
                        new_params.pop('click_x', None)
                        new_params.pop('click_y', None)
                        new_params.pop('image_key', None)
                        new_params.pop('timestamp', None)
                        
                        # クエリパラメータを更新
                        st.query_params.clear()
                        for key, value in new_params.items():
                            if isinstance(value, list):
                                for v in value:
                                    st.query_params[key] = v
                            else:
                                st.query_params[key] = value
                        
                        print(f"[DEBUG] リロードを実行します")
                        st.rerun()
                    else:
                        print(f"[DEBUG] このクリックは既に処理済みです")
                else:
                    st.warning(f"⚠️ 座標が画像の範囲外です: ({click_x}, {click_y})")
                    print(f"[DEBUG] 座標が範囲外: ({click_x}, {click_y})")
            except (ValueError, TypeError) as e:
                st.error(f"座標の変換エラー: {e}")
                print(f"[DEBUG] 座標変換エラー: {e}")
                import traceback
                print(f"[DEBUG] トレースバック: {traceback.format_exc()}")
        else:
            print(f"[DEBUG] image_keyが一致しません: 受信={click_image_key}, 期待={image_key}")
    else:
        # デバッグ: URLパラメータに必要なキーがない場合
        if query_params:
            print(f"[DEBUG] URLパラメータに必要なキーがありません。現在のキー: {list(query_params.keys())}")
    
    # 画像情報を表示
    st.info(f"📐 画像サイズ: 幅 {image.width}px × 高さ {image.height}px")
    
    # クリック状態を表示
    click_count = st.session_state[f'click_count_{image_key}']
    if click_count % 2 == 0:
        st.info("🖱️ **画像をクリックして左上の点を選択してください**")
    else:
        st.info("🖱️ **画像をクリックして右下の点を選択してください**")
    
    # 2カラムレイアウト
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 画像を表示（カーソル位置の座標を表示）
        display_image = image.copy()
        
        # 現在選択中の2点を描画
        if current_points['top_left'] is not None:
            x, y = current_points['top_left']
            display_image = draw_point_on_image(display_image, x, y, (255, 0, 0), size=10)  # 赤
        
        if current_points['bottom_right'] is not None:
            x, y = current_points['bottom_right']
            display_image = draw_point_on_image(display_image, x, y, (0, 255, 0), size=10)  # 緑
        
        # 2点が選択されている場合は矩形を描画
        if current_points['top_left'] is not None and current_points['bottom_right'] is not None:
            x1, y1 = current_points['top_left']
            x2, y2 = current_points['bottom_right']
            
            # OpenCVが利用可能な場合のみ矩形を描画
            if CV2_AVAILABLE and cv2 is not None:
                # 矩形を描画
                img_array = np.array(display_image)
                if len(img_array.shape) == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array
                
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), 2)  # マゼンタ色
                
                if len(img_bgr.shape) == 3:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img_bgr
                
                display_image = Image.fromarray(img_rgb)
        
        # 画像を表示（クリックで座標を取得できるようにst.components.v1.htmlを使用）
        # display_imageがPIL Imageであることを確認し、確実にPIL Imageに変換
        try:
            # 既にPIL Imageの場合はそのまま使用
            if isinstance(display_image, Image.Image):
                final_display_image = display_image
            elif isinstance(display_image, np.ndarray):
                # numpy配列の場合はPIL Imageに変換
                if len(display_image.shape) == 3:
                    # BGRからRGBに変換（OpenCV形式の場合）
                    if CV2_AVAILABLE and cv2 is not None:
                        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                    final_display_image = Image.fromarray(display_image)
                else:
                    final_display_image = Image.fromarray(display_image)
            else:
                # その他の型の場合はエラー
                raise TypeError(f"Unsupported image type: {type(display_image)}")
            
            # PIL Imageを確実にPIL Imageとして扱う
            if not isinstance(final_display_image, Image.Image):
                raise TypeError(f"Failed to convert to PIL Image: {type(final_display_image)}")
            
            # PIL ImageをRGBモードに変換（Streamlit Cloudでの互換性のため）
            if final_display_image.mode != 'RGB':
                final_display_image = final_display_image.convert('RGB')
            
            # 画像サイズを適切にリサイズ（表示用）
            # 大きすぎる画像は縮小して表示（最大幅1200px、アスペクト比を保持）
            max_display_width = 1200
            max_display_height = 800
            
            display_width = final_display_image.width
            display_height = final_display_image.height
            
            # リサイズが必要かチェック
            if display_width > max_display_width or display_height > max_display_height:
                # アスペクト比を保持してリサイズ
                scale = min(max_display_width / display_width, max_display_height / display_height)
                display_width = int(display_width * scale)
                display_height = int(display_height * scale)
                display_image_resized = final_display_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            else:
                display_image_resized = final_display_image
                scale = 1.0
            
            # スケールファクターをセッション状態に保存（座標変換用）
            st.session_state[f'image_scale_{image_key}'] = scale
            st.session_state[f'original_image_size_{image_key}'] = (final_display_image.width, final_display_image.height)
            
            # 選択された点を画像上に描画（PILを使用）
            display_img_with_points = display_image_resized.copy()
            from PIL import ImageDraw
            
            if current_points['top_left'] is not None:
                x, y = current_points['top_left']
                # 表示用画像の座標に変換
                display_x = int(x * scale) if scale != 1.0 else x
                display_y = int(y * scale) if scale != 1.0 else y
                # 点を描画（赤色の円）
                draw = ImageDraw.Draw(display_img_with_points)
                draw.ellipse([display_x - 8, display_y - 8, display_x + 8, display_y + 8], fill=(255, 0, 0), outline=(255, 0, 0), width=2)
            
            if current_points['bottom_right'] is not None:
                x, y = current_points['bottom_right']
                # 表示用画像の座標に変換
                display_x = int(x * scale) if scale != 1.0 else x
                display_y = int(y * scale) if scale != 1.0 else y
                # 点を描画（緑色の円）
                draw = ImageDraw.Draw(display_img_with_points)
                draw.ellipse([display_x - 8, display_y - 8, display_x + 8, display_y + 8], fill=(0, 255, 0), outline=(0, 255, 0), width=2)
            
            # 2点が選択されている場合は矩形を描画
            if current_points['top_left'] is not None and current_points['bottom_right'] is not None:
                x1, y1 = current_points['top_left']
                x2, y2 = current_points['bottom_right']
                # 表示用画像の座標に変換
                display_x1 = int(x1 * scale) if scale != 1.0 else x1
                display_y1 = int(y1 * scale) if scale != 1.0 else y1
                display_x2 = int(x2 * scale) if scale != 1.0 else x2
                display_y2 = int(y2 * scale) if scale != 1.0 else y2
                # 矩形を描画（マゼンタ色）
                draw = ImageDraw.Draw(display_img_with_points)
                draw.rectangle([display_x1, display_y1, display_x2, display_y2], outline=(255, 0, 255), width=2)
            
            # streamlit-drawable-canvasを使用してクリック座標を取得
            try:
                from streamlit_drawable_canvas import st_canvas
                
                st.markdown("**🖱️ 画像をクリックして座標を選択してください**")
                st.caption("1回目のクリック: 左上の点、2回目のクリック: 右下の点")
                
                # カーソル位置の座標表示（st.components.v1.htmlを使用）
                try:
                    if hasattr(st.components, 'v1') and hasattr(st.components.v1, 'html'):
                        html_content = create_image_with_coord_display(
                            display_img_with_points, 
                            image_key,
                            original_width=final_display_image.width,
                            original_height=final_display_image.height
                        )
                        # 高さを適切に設定（画像の高さ + 余白）
                        display_height_html = min(display_height + 100, 1000)
                        
                        if display_height_html <= 0:
                            display_height_html = 600  # デフォルト値
                        
                        st.components.v1.html(html_content, height=display_height_html, scrolling=False)
                except Exception as html_error:
                    # カーソル座標表示が失敗しても続行
                    pass
                
                # 前回のクリック数を取得（重複処理を防ぐため）
                last_click_count_key = f'last_click_count_{image_key}'
                if last_click_count_key not in st.session_state:
                    st.session_state[last_click_count_key] = 0
                
                # streamlit-drawable-canvasでクリック座標を取得
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",  # 塗りつぶし色（赤、半透明）
                    stroke_width=2,
                    stroke_color="#FF0000",  # 線の色（赤）
                    background_image=display_img_with_points,
                    update_streamlit=True,  # クリックを検出するためにTrueに設定
                    height=display_height,
                    width=display_width,
                    drawing_mode="point",  # ポイントモードでクリックを検出
                    point_display_radius=5,  # ポイントの表示半径
                    key=f"canvas_{image_key}",
                )
                
                # クリックされた座標を取得
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    current_click_count = len(objects)
                    
                    print(f"[DEBUG] canvas_result.json_data: {canvas_result.json_data}")
                    print(f"[DEBUG] objects: {objects}")
                    print(f"[DEBUG] current_click_count: {current_click_count}, last_click_count: {st.session_state[last_click_count_key]}")
                    
                    # クリック数が増えた場合のみ処理（重複処理を防ぐ）
                    if current_click_count > st.session_state[last_click_count_key]:
                        if objects:
                            # 最新の2つのポイントを取得
                            # 表示用画像の座標を元の画像座標に変換
                            points = []
                            for obj in objects[-2:]:
                                # 表示用画像の座標
                                display_x = int(obj.get("left", 0))
                                display_y = int(obj.get("top", 0))
                                # 元の画像座標に変換
                                orig_x = int(display_x / scale) if scale != 1.0 else display_x
                                orig_y = int(display_y / scale) if scale != 1.0 else display_y
                                points.append((orig_x, orig_y))
                                print(f"[DEBUG] 座標変換: 表示({display_x}, {display_y}) -> 元({orig_x}, {orig_y}), scale={scale}")
                            
                            if len(points) >= 1:
                                # 1回目のクリック: 左上の点
                                current_points['top_left'] = points[0]
                                st.session_state[f'click_count_{image_key}'] = 1
                                print(f"[DEBUG] 左上の点を設定: {points[0]}")
                                
                                if len(points) >= 2:
                                    # 2回目のクリック: 右下の点
                                    current_points['bottom_right'] = points[1]
                                    st.session_state[f'click_count_{image_key}'] = 2
                                    print(f"[DEBUG] 右下の点を設定: {points[1]}")
                                
                                # セッション状態を更新
                                st.session_state[f'current_points_{image_key}'] = current_points
                                
                                # 数値入力フィールドのセッション状態も更新
                                if current_points['top_left']:
                                    st.session_state[f'top_left_x_{image_key}'] = current_points['top_left'][0]
                                    st.session_state[f'top_left_y_{image_key}'] = current_points['top_left'][1]
                                if current_points['bottom_right']:
                                    st.session_state[f'bottom_right_x_{image_key}'] = current_points['bottom_right'][0]
                                    st.session_state[f'bottom_right_y_{image_key}'] = current_points['bottom_right'][1]
                                
                                # クリック数を更新
                                st.session_state[last_click_count_key] = current_click_count
                                
                                # 成功メッセージを表示
                                if len(points) == 1:
                                    st.success(f"✅ 左上の点を選択しました: ({points[0][0]}, {points[0][1]})")
                                elif len(points) >= 2:
                                    st.success(f"✅ 右下の点を選択しました: ({points[1][0]}, {points[1][1]})")
                                
                                # リロードはst_canvasのupdate_streamlitで自動的に行われる
                                
            except ImportError:
                # streamlit-drawable-canvasがインストールされていない場合
                st.warning("⚠️ streamlit-drawable-canvasがインストールされていません。数値入力フィールドで座標を指定してください。")
                st.info("💡 インストール方法: `pip install streamlit-drawable-canvas` または `uv pip install streamlit-drawable-canvas`")
                
                # st.imageに渡す（Streamlit Cloudの古いバージョンではuse_column_widthを使用）
                try:
                    st.image(display_img_with_points, caption="画像プレビュー（座標は数値入力フィールドで指定してください）", use_container_width=True)
                except TypeError:
                    st.image(display_img_with_points, caption="画像プレビュー（座標は数値入力フィールドで指定してください）", use_column_width=True)
            except Exception as canvas_error:
                # streamlit-drawable-canvasでエラーが発生した場合
                st.warning("⚠️ クリック座標取得機能でエラーが発生しました。数値入力フィールドで座標を指定してください。")
                
                # エラーの詳細を表示（デバッグ用）
                import traceback
                error_details = traceback.format_exc()
                with st.expander("エラー詳細（デバッグ用）", expanded=False):
                    st.code(error_details)
                
                # st.imageに渡す（Streamlit Cloudの古いバージョンではuse_column_widthを使用）
                try:
                    st.image(display_img_with_points, caption="画像プレビュー（座標は数値入力フィールドで指定してください）", use_container_width=True)
                except TypeError:
                    st.image(display_img_with_points, caption="画像プレビュー（座標は数値入力フィールドで指定してください）", use_column_width=True)
        except Exception as e:
            st.error(f"画像表示エラー: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.info("画像の表示に失敗しました。数値入力フィールドで座標を指定してください。")
        
        # 範囲が登録されている場合は可視化した画像も表示
        if regions:
            visualized_image = visualize_regions_on_image(image, regions)
            try:
                st.image(visualized_image, caption="登録済み範囲", use_container_width=True)
            except TypeError:
                # 古いStreamlitバージョンではuse_column_widthを使用
                st.image(visualized_image, caption="登録済み範囲", use_column_width=True)
    
    with col2:
        st.subheader("2点の座標を入力")
        
        # OpenCVウィンドウで座標を選択するボタン
        if st.button("🖱️ OpenCVウィンドウで座標を選択", key=f"opencv_picker_{image_key}", type="secondary"):
            try:
                st.info("OpenCVウィンドウが開きます。画像上で左上と右下の2点をクリックし、Enterキーで確定してください。ESCキーで終了します。")
                st.warning("⚠️ 注意: OpenCVウィンドウはサーバー側で開きます。ローカルで実行している場合のみ表示されます。")
                
                # OpenCVウィンドウを開いて座標を取得
                # 注意: StreamlitはWebアプリなので、OpenCVウィンドウはサーバー側で開かれます
                # ローカルで実行している場合（streamlit run）のみ、ウィンドウが表示されます
                coords_list = open_opencv_coord_picker(image, image_key)
                
                if coords_list and len(coords_list) > 0:
                    # 取得したすべての範囲をregionsに追加
                    for coord_dict in coords_list:
                        if coord_dict and coord_dict.get('top_left') and coord_dict.get('bottom_right'):
                            x1, y1 = coord_dict['top_left']
                            x2, y2 = coord_dict['bottom_right']
                            
                            # 座標を正規化（左上が小さい値、右下が大きい値になるように）
                            x1, x2 = min(x1, x2), max(x1, x2)
                            y1, y2 = min(y1, y2), max(y1, y2)
                            
                            if x1 < x2 and y1 < y2:
                                regions.append({
                                    'coords': (int(x1), int(y1), int(x2), int(y2)),
                                    'name': f'{len(regions) + 1}'
                                })
                    
                    st.session_state[f'click_regions_{image_key}'] = regions
                    st.success(f"{len(coords_list)} 個の範囲を取得しました！")
                    st.rerun()
                else:
                    st.warning("座標が取得できませんでした。OpenCVウィンドウが表示されていない可能性があります。")
            except RuntimeError as e:
                error_msg = str(e)
                if "GUIサポート" in error_msg or "not implemented" in error_msg.lower():
                    st.error("⚠️ OpenCVのGUIサポートが利用できません")
                    st.warning("この環境ではOpenCVウィンドウを使用できません。数値入力フィールドで座標を手動入力してください。")
                    with st.expander("🔧 解決方法", expanded=True):
                        st.markdown("""
                        **OpenCVのGUIサポートを有効にするには：**
                        
                        1. **opencv-python-headlessをアンインストール**（インストールされている場合）
                           ```bash
                           pip uninstall opencv-python-headless
                           ```
                          または
                           ```bash
                           uv pip uninstall opencv-python-headless
                           ```
                        
                        2. **opencv-pythonを再インストール**
                           ```bash
                           pip install --force-reinstall opencv-python
                           ```
                          または
                           ```bash
                           uv pip install --force-reinstall opencv-python
                           ```
                        
                        3. **アプリを再起動**
                        """)
                else:
                    st.error(f"OpenCVウィンドウのエラー: {e}")
                    st.info("💡 ヒント: OpenCVウィンドウはローカルで実行している場合のみ表示されます。")
            except Exception as e:
                error_msg = str(e)
                if "not implemented" in error_msg.lower() or "gtk" in error_msg.lower():
                    st.error("⚠️ OpenCVのGUIサポートが利用できません")
                    st.warning("この環境ではOpenCVウィンドウを使用できません。数値入力フィールドで座標を手動入力してください。")
                    with st.expander("🔧 解決方法", expanded=True):
                        st.markdown("""
                        **OpenCVのGUIサポートを有効にするには：**
                        
                        1. **opencv-python-headlessをアンインストール**（インストールされている場合）
                           ```bash
                           pip uninstall opencv-python-headless
                           ```
                        
                        2. **opencv-pythonを再インストール**
                           ```bash
                           pip install --force-reinstall opencv-python
                           ```
                        
                        3. **アプリを再起動**
                        """)
                else:
                    st.error(f"OpenCVウィンドウのエラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.markdown("**または手動で入力**: 画像上でマウスを動かして座標を確認し、数値入力フィールドで座標を入力してください")
        
        # 左上の点
        st.markdown("**1. 左上の点** 🔴")
        col_x1, col_y1 = st.columns(2)
        with col_x1:
            # セッション状態から値を取得（セッション状態に値がない場合はcurrent_pointsから取得）
            if f'top_left_x_{image_key}' in st.session_state:
                top_left_x_value = st.session_state[f'top_left_x_{image_key}']
            else:
                top_left_x_value = current_points['top_left'][0] if current_points['top_left'] else 0
            top_left_x = st.number_input("X1", min_value=0, max_value=image.width,
                                         value=top_left_x_value,
                                         key=f"top_left_x_{image_key}")
        with col_y1:
            # セッション状態から値を取得（セッション状態に値がない場合はcurrent_pointsから取得）
            if f'top_left_y_{image_key}' in st.session_state:
                top_left_y_value = st.session_state[f'top_left_y_{image_key}']
            else:
                top_left_y_value = current_points['top_left'][1] if current_points['top_left'] else 0
            top_left_y = st.number_input("Y1", min_value=0, max_value=image.height,
                                         value=top_left_y_value,
                                         key=f"top_left_y_{image_key}")
        
        # 右下の点
        st.markdown("**2. 右下の点** 🟢")
        col_x2, col_y2 = st.columns(2)
        with col_x2:
            # セッション状態から値を取得（セッション状態に値がない場合はcurrent_pointsから取得）
            if f'bottom_right_x_{image_key}' in st.session_state:
                bottom_right_x_value = st.session_state[f'bottom_right_x_{image_key}']
            else:
                bottom_right_x_value = current_points['bottom_right'][0] if current_points['bottom_right'] else image.width
            bottom_right_x = st.number_input("X2", min_value=0, max_value=image.width,
                                            value=bottom_right_x_value,
                                            key=f"bottom_right_x_{image_key}")
        with col_y2:
            # セッション状態から値を取得（セッション状態に値がない場合はcurrent_pointsから取得）
            if f'bottom_right_y_{image_key}' in st.session_state:
                bottom_right_y_value = st.session_state[f'bottom_right_y_{image_key}']
            else:
                bottom_right_y_value = current_points['bottom_right'][1] if current_points['bottom_right'] else image.height
            bottom_right_y = st.number_input("Y2", min_value=0, max_value=image.height,
                                            value=bottom_right_y_value,
                                            key=f"bottom_right_y_{image_key}")
        
        # 数値入力フィールドの値がセッション状態と異なる場合は更新
        new_top_left = (int(top_left_x), int(top_left_y))
        new_bottom_right = (int(bottom_right_x), int(bottom_right_y))
        
        # 座標が変更された場合はcurrent_pointsのみを更新
        # 注意: st.number_inputでkeyを指定した後は、そのキーに対応するセッション状態を直接変更できない
        if (current_points['top_left'] != new_top_left or 
            current_points['bottom_right'] != new_bottom_right):
            current_points['top_left'] = new_top_left
            current_points['bottom_right'] = new_bottom_right
            st.session_state[f'current_points_{image_key}'] = current_points
            # 数値入力フィールドの値が変更されると自動的にページが再読み込みされるため、
            # 画像も自動的に更新される
        
        # 範囲を確定するボタン
        if st.button("✅ 範囲を確定", key=f"confirm_region_{image_key}", type="primary"):
            if current_points['top_left'] is not None and current_points['bottom_right'] is not None:
                x1, y1 = current_points['top_left']
                x2, y2 = current_points['bottom_right']
                
                # 座標を正規化（左上が小さい値、右下が大きい値になるように）
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                if x1 < x2 and y1 < y2:
                    regions.append({
                        'coords': (int(x1), int(y1), int(x2), int(y2)),
                        'name': f'{len(regions) + 1}'
                    })
                    st.session_state[f'click_regions_{image_key}'] = regions
                    # 2点をリセット
                    st.session_state[f'current_points_{image_key}'] = {
                        'top_left': None,
                        'bottom_right': None
                    }
                    # 数値入力フィールド用のセッション状態キーもクリア
                    if f'top_left_x_{image_key}' in st.session_state:
                        del st.session_state[f'top_left_x_{image_key}']
                    if f'top_left_y_{image_key}' in st.session_state:
                        del st.session_state[f'top_left_y_{image_key}']
                    if f'bottom_right_x_{image_key}' in st.session_state:
                        del st.session_state[f'bottom_right_x_{image_key}']
                    if f'bottom_right_y_{image_key}' in st.session_state:
                        del st.session_state[f'bottom_right_y_{image_key}']
                    st.session_state[f'click_count_{image_key}'] = 0
                    st.session_state[f'processed_clicks_{image_key}'] = set()
                    st.success("範囲を追加しました")
                    st.rerun()
                else:
                    st.error("座標が無効です。有効な矩形を選択してください。")
            else:
                st.warning("左上と右下の2点を選択してください。")
        
        # 2点をリセットするボタン
        if st.button("🔄 2点をリセット", key=f"reset_points_{image_key}"):
            st.session_state[f'current_points_{image_key}'] = {
                'top_left': None,
                'bottom_right': None
            }
            # 数値入力フィールド用のセッション状態キーもクリア
            if f'top_left_x_{image_key}' in st.session_state:
                del st.session_state[f'top_left_x_{image_key}']
            if f'top_left_y_{image_key}' in st.session_state:
                del st.session_state[f'top_left_y_{image_key}']
            if f'bottom_right_x_{image_key}' in st.session_state:
                del st.session_state[f'bottom_right_x_{image_key}']
            if f'bottom_right_y_{image_key}' in st.session_state:
                del st.session_state[f'bottom_right_y_{image_key}']
            st.session_state[f'click_count_{image_key}'] = 0
            st.session_state[f'processed_clicks_{image_key}'] = set()
            st.rerun()
        
        # 既存の範囲を表示・削除
        if regions:
            st.subheader("登録済み範囲")
            for i, region in enumerate(regions):
                with st.expander(f"📦 {region['name']}", expanded=False):
                    coords = region['coords']
                    st.write(f"**座標**: ({coords[0]}, {coords[1]}) - ({coords[2]}, {coords[3]})")
                    st.write(f"**サイズ**: 幅 {coords[2] - coords[0]}px × 高さ {coords[3] - coords[1]}px")
                    
                    if st.button("🗑️ 削除", key=f"delete_{i}_{image_key}"):
                        regions.pop(i)
                        st.session_state[f'click_regions_{image_key}'] = regions
                        st.rerun()
    
    return regions


def process_files(
    files: List,
    regions: List[Dict],
    pages: Optional[List[int]] = None
) -> List[Dict]:
    """
    ファイルを処理してテキストを抽出
    
    Args:
        files: アップロードされたファイルのリスト
        regions: 抽出領域のリスト
        pages: PDFの場合のページ番号リスト
    
    Returns:
        抽出結果のリスト
    """
    results = []
    
    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in files:
            try:
                # ファイルを一時保存
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # ファイルを処理
                file_results = st.session_state.extractor.process_file(
                    file_path, regions, pages
                )
                
                # 結果をリストに追加（PDFの場合は複数ページ、画像の場合は1つ）
                if isinstance(file_results, list):
                    results.extend(file_results)
                else:
                    results.append(file_results)
            
            except Exception as e:
                results.append({
                    'filename': uploaded_file.name,
                    'error': str(e)
                })
    
    return results


def export_to_excel(results: List[Dict], regions: List[Dict]) -> bytes:
    """
    結果をExcelファイルにエクスポート
    
    Args:
        results: 抽出結果のリスト
        regions: 抽出領域のリスト（列名の順序を決定）
    
    Returns:
        Excelファイルのバイトデータ
    """
    # データフレームを作成
    rows = []
    
    for result in results:
        row = {}
        
        # 基本情報
        row['ファイル名'] = result.get('filename', '')
        if 'page' in result:
            row['ページ'] = result.get('page', '')
        
        # 各領域のテキスト
        region_names = [r.get('name', f'{i+1}') for i, r in enumerate(regions)]
        for name in region_names:
            row[name] = result.get(name, '')
        
        # エラー情報
        if 'error' in result:
            row['エラー'] = result.get('error', '')
        
        rows.append(row)
    
    # データフレームを作成
    df = pd.DataFrame(rows)
    
    # Excelに変換
    # pandasのto_excelは既にUTF-8でエンコードされているが、
    # 念のため列名が正しくエンコードされていることを確認
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='抽出結果')
    
    output.seek(0)
    return output.getvalue()


def main():
    """メインアプリケーション"""
    
    st.title("📄 Scan To Sheet - OCR抽出ツール")
    st.markdown("画像・PDFからドラッグで選択した範囲のテキストを抽出します")
    
    # サイドバー
    with st.sidebar:
        st.header("設定")
        
        # 日本語データの状態をチェック（Tesseractが利用可能な場合のみ）
        tesseract_path = get_tesseract_path()
        tessdata_path = get_tessdata_path(tesseract_path) if tesseract_path else None
        has_jpn_data = check_japanese_data(tessdata_path) if tessdata_path else False
        
        # Tesseractが利用可能で、日本語データがない場合のみ警告を表示
        if tesseract_path and not has_jpn_data:
            st.error("⚠️ 日本語データが見つかりません")
            with st.expander("📋 インストール手順", expanded=True):
                st.markdown("""
                **日本語OCRを使用するには、以下の手順で日本語データをインストールしてください：**
                
                1. **ダウンロード**
                   - [jpn.traineddata](https://github.com/tesseract-ocr/tessdata/raw/main/jpn.traineddata) をダウンロード
                
                2. **配置場所**
                """)
                jpn_data_path = os.path.join(os.path.dirname(tesseract_path), 'tessdata', 'jpn.traineddata')
                st.code(jpn_data_path, language=None)
                
                st.markdown("""
                3. **再起動**
                   - ファイルを配置した後、アプリを再起動してください
                
                **💡 ヒント**: EasyOCRを使用すると、日本語データのインストール不要で日本語OCRが利用できます。
                """)
        elif tesseract_path and has_jpn_data:
            st.success("✓ 日本語データが利用可能です")
        elif not tesseract_path:
            # Tesseractが利用できない環境（Streamlit Cloudなど）では、EasyOCRを推奨
            if EASYOCR_AVAILABLE:
                st.info("💡 **EasyOCRを使用中**: 日本語OCRが利用可能です（追加設定不要）")
            else:
                st.warning("⚠️ Tesseractが利用できません。EasyOCRのインストールを推奨します。")
        
        # OCRエンジン設定
        st.subheader("OCRエンジン設定")
        ocr_engine_options = []
        if EASYOCR_AVAILABLE:
            ocr_engine_options.append('EasyOCR (AI搭載・高精度・推奨)')
        else:
            ocr_engine_options.append('EasyOCR (AI搭載・高精度) - 未インストール')
        
        # Tesseractが利用可能な場合のみオプションに追加
        tesseract_path = get_tesseract_path()
        if tesseract_path:
            ocr_engine_options.append('Tesseract (標準)')
        
        # 現在のエンジンに応じてインデックスを設定
        current_engine_index = 0
        if st.session_state.ocr_engine == 'tesseract':
            if tesseract_path and EASYOCR_AVAILABLE:
                current_engine_index = 1  # Tesseractが2番目
            elif not EASYOCR_AVAILABLE and tesseract_path:
                current_engine_index = 0  # Tesseractのみ
            else:
                # Tesseractが利用できない場合はEasyOCRに強制
                if EASYOCR_AVAILABLE:
                    st.session_state.ocr_engine = 'easyocr'
                    current_engine_index = 0
                else:
                    st.error("⚠️ OCRエンジンが利用できません。")
        elif st.session_state.ocr_engine == 'easyocr':
            if EASYOCR_AVAILABLE:
                current_engine_index = 0  # EasyOCRが1番目
            else:
                # EasyOCRが利用できない場合はTesseractに戻す（利用可能な場合）
                if tesseract_path:
                    st.session_state.ocr_engine = 'tesseract'
                    current_engine_index = 0
        
        selected_engine_display = st.selectbox(
            "OCRエンジン",
            options=ocr_engine_options,
            index=current_engine_index,
            help="EasyOCRは低解像度画像に強いAI搭載エンジンです。初回使用時にモデルをダウンロードします（約500MB）。"
        )
        
        # 選択されたエンジンをセッション状態に保存
        if 'EasyOCR' in selected_engine_display and EASYOCR_AVAILABLE:
            new_engine = 'easyocr'
        else:
            new_engine = 'tesseract'
        
        # エンジンが変更された場合、新しいextractorを作成
        if st.session_state.ocr_engine != new_engine:
            st.session_state.ocr_engine = new_engine
            # 既存の言語設定を保持
            current_lang = st.session_state.extractor.lang
            st.session_state.extractor = OCRExtractor(lang=current_lang, ocr_engine=new_engine)
        
        # EasyOCRが未インストールの場合の警告
        if 'EasyOCR' in selected_engine_display and not EASYOCR_AVAILABLE:
            st.warning("⚠️ EasyOCRがインストールされていません。")
            st.info("💡 EasyOCRをインストールするには: `pip install easyocr` または `uv pip install easyocr`")
            st.session_state.ocr_engine = 'tesseract'
            if st.session_state.extractor.ocr_engine != 'tesseract':
                current_lang = st.session_state.extractor.lang
                st.session_state.extractor = OCRExtractor(lang=current_lang, ocr_engine='tesseract')
        
        # OCR言語設定
        lang_options = {
            '日本語+英語': 'eng+jpn',  # 英語を優先（数字の誤認識を防ぐため）
            '英語のみ': 'eng',
            '日本語のみ': 'jpn'
        }
        selected_lang = st.selectbox(
            "OCR言語",
            options=list(lang_options.keys()),
            index=0
        )
        # jpn+engが指定された場合はeng+jpnに変換（後方互換性のため）
        lang_value = lang_options[selected_lang]
        if lang_value == 'jpn+eng':
            lang_value = 'eng+jpn'
        
        # 言語設定が変更された場合、extractorを更新
        if st.session_state.extractor.lang != lang_value:
            st.session_state.extractor.lang = lang_value
            # EasyOCRを使用している場合は、言語変更に応じて再初期化が必要な場合がある
            # ただし、EasyOCRは実行時に言語を変更できるため、ここではlangのみ更新
        
        # 日本語データがない場合は警告を表示（Tesseract使用時のみ、かつTesseractが利用可能な場合）
        if (st.session_state.ocr_engine == 'tesseract' and tesseract_path and 
            not has_jpn_data and selected_lang in ['日本語+英語', '日本語のみ']):
            st.warning("⚠️ 日本語データがインストールされていないため、英語のみで認識されます。")
            if EASYOCR_AVAILABLE:
                st.info("💡 EasyOCRに切り替えると、日本語データのインストール不要で日本語OCRが利用できます。")
        
        # PDF処理設定
        st.subheader("PDF設定")
        use_text_layer = st.checkbox("テキストレイヤーを優先", value=True)
        
        # クリアボタン
        if st.button("すべてクリア", type="secondary"):
            st.session_state.uploaded_files = []
            st.session_state.selected_regions = []
            st.session_state.processing_results = []
            st.session_state.current_file_index = 0
            st.session_state.current_image = None
            st.session_state.selected_files_for_processing = []
            st.rerun()
    
    # メインコンテンツ
    tab1, tab2, tab3 = st.tabs(["📤 ファイルアップロード", "🎯 範囲選択", "📊 結果表示・エクスポート"])
    
    with tab1:
        st.header("ファイルをアップロード")
        
        uploaded_files = st.file_uploader(
            "画像またはPDFファイルを選択",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"{len(uploaded_files)} 個のファイルがアップロードされました")
            
            # ファイル一覧を表示
            st.subheader("アップロードされたファイル")
            for i, file in enumerate(uploaded_files):
                st.write(f"{i + 1}. {file.name} ({file.size / 1024:.1f} KB)")
    
    with tab2:
        st.header("抽出範囲を選択")
        
        if not st.session_state.uploaded_files:
            st.info("まず「ファイルアップロード」タブでファイルをアップロードしてください")
        else:
            # 複数ファイル選択（チェックボックス形式）
            st.subheader("処理するファイルを選択（複数選択可能）")
            
            # セッション状態を初期化
            if 'file_selection_checkboxes' not in st.session_state:
                st.session_state.file_selection_checkboxes = {}
            
            # 各ファイルに対してチェックボックスを表示
            selected_file_names = []
            file_names = [f.name for f in st.session_state.uploaded_files]
            
            # 既存の選択状態を初期化（新しくアップロードされたファイルがある場合）
            for file_name in file_names:
                if file_name not in st.session_state.file_selection_checkboxes:
                    # 以前の選択状態があれば継承
                    if file_name in (st.session_state.selected_files_for_processing or []):
                        st.session_state.file_selection_checkboxes[file_name] = True
                    else:
                        st.session_state.file_selection_checkboxes[file_name] = False
            
            # チェックボックスを表示
            cols = st.columns(3)  # 3列に分割して表示
            for idx, file_name in enumerate(file_names):
                col_idx = idx % 3
                with cols[col_idx]:
                    checked = st.checkbox(
                        file_name,
                        value=st.session_state.file_selection_checkboxes.get(file_name, False),
                        key=f"file_checkbox_{file_name}"
                    )
                    st.session_state.file_selection_checkboxes[file_name] = checked
                    if checked:
                        selected_file_names.append(file_name)
            
            # セッション状態を更新
            st.session_state.selected_files_for_processing = selected_file_names
            
            # 範囲選択用の代表ファイルを選択（最初に選択したファイル、または明示的に選択）
            representative_file = None
            representative_file_name = None
            
            if selected_file_names:
                # 代表ファイルは最初に選択したファイルを使用
                representative_file_name = selected_file_names[0]
                for f in st.session_state.uploaded_files:
                    if f.name == representative_file_name:
                        representative_file = f
                        break
                
                # 選択されたファイル数を表示
                st.info(f"📁 {len(selected_file_names)} 個のファイルが選択されています。範囲選択は代表ファイル（{representative_file_name}）で行います。")
            
            if representative_file:
                # ファイルタイプを判定
                file_ext = Path(representative_file.name).suffix.lower()
                is_pdf = file_ext == '.pdf'
                st.session_state.current_file_type = 'pdf' if is_pdf else 'image'
                
                if is_pdf:
                    # PDFの場合
                    st.subheader("PDFページ選択")
                    st.caption("💡 ページ設定は代表ファイルの設定を使用します。すべての選択PDFファイルに同じ設定が適用されます。")
                    
                    # ページ数を取得（簡易版：最初の10ページまで）
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(representative_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        if PYMUPDF_AVAILABLE:
                            import fitz
                            doc = fitz.open(tmp_path)
                            total_pages = len(doc)
                            doc.close()
                        else:
                            total_pages = 10  # デフォルト値
                        
                        os.unlink(tmp_path)
                    except:
                        total_pages = 10
                    
                    page_option = st.radio(
                        "処理するページ",
                        options=["全ページ", "特定のページ"],
                        horizontal=True
                    )
                    
                    selected_pages = None
                    if page_option == "特定のページ":
                        page_numbers = st.multiselect(
                            "ページ番号を選択（1始まり）",
                            options=list(range(1, total_pages + 1)),
                            default=[1]
                        )
                        if page_numbers:
                            selected_pages = [p - 1 for p in page_numbers]  # 0始まりに変換
                    
                    # PDFの最初のページを画像として表示
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(representative_file.getbuffer())
                            tmp_path = tmp_file.name
                        
                        images = pdf_to_images_from_path(tmp_path, pages=[0])
                        if images:
                            st.session_state.current_image = images[0]
                            display_image = convert_image_for_display(images[0])
                            os.unlink(tmp_path)
                        else:
                            st.error("PDFの読み込みに失敗しました")
                            display_image = None
                    except Exception as e:
                        st.error(f"PDF処理エラー: {e}")
                        display_image = None
                else:
                    # 画像ファイルの場合
                    try:
                        image_bytes = representative_file.getbuffer()
                        image = bytes_to_image(image_bytes)
                        st.session_state.current_image = image
                        display_image = convert_image_for_display(image)
                        selected_pages = None
                    except Exception as e:
                        st.error(f"画像読み込みエラー: {e}")
                        display_image = None
                
                if display_image:
                    st.subheader("2点の座標を入力して範囲を選択")
                    st.caption(f"💡 範囲選択は代表ファイル（{representative_file_name}）で行います。選択した範囲はすべての選択ファイルに適用されます。")
                    
                    # 2点座標入力UIを表示（代表ファイルの名前を使用）
                    regions = render_click_coord_input(display_image, representative_file_name)
                    st.session_state.selected_regions = regions
                    
                    if regions:
                        st.success(f"{len(regions)} 個の範囲が登録されました")
                        
                        # 選択されたファイルの一覧を表示
                        if len(selected_file_names) > 1:
                            st.subheader("選択されたファイル")
                            for i, file_name in enumerate(selected_file_names, 1):
                                st.write(f"{i}. {file_name}")
                        
                        # 処理ボタン
                        if st.button("📝 テキスト抽出を実行", type="primary"):
                            if not selected_file_names:
                                st.warning("処理するファイルを選択してください。")
                            else:
                                with st.spinner(f"処理中... ({len(selected_file_names)} 個のファイル)"):
                                    # 選択されたすべてのファイルを取得
                                    selected_files = []
                                    for file_name in selected_file_names:
                                        for f in st.session_state.uploaded_files:
                                            if f.name == file_name:
                                                selected_files.append(f)
                                                break
                                    
                                    # すべてのファイルを処理
                                    # PDFの場合は全ページを処理（selected_pagesは代表ファイルの設定を使用）
                                    results = process_files(
                                        selected_files,
                                        regions,
                                        selected_pages  # PDFの場合は代表ファイルのページ設定を使用
                                    )
                                    
                                    st.session_state.processing_results.extend(results)
                                    st.success(f"処理が完了しました！{len(selected_files)} 個のファイルを処理しました。「結果表示・エクスポート」タブを確認してください。")
                                    st.rerun()
                    else:
                        st.info("2点の座標を入力して範囲を追加してください")
            elif selected_file_names:
                st.warning("代表ファイルの読み込みに失敗しました。別のファイルを選択してください。")
    
    with tab3:
        st.header("抽出結果")
        
        if not st.session_state.processing_results:
            st.info("まだ処理結果がありません。「範囲選択」タブでテキスト抽出を実行してください。")
        else:
            # 結果を表示
            st.subheader("抽出結果一覧")
            
            # データフレームとして表示
            display_data = []
            for result in st.session_state.processing_results:
                row = {
                    'ファイル名': result.get('filename', ''),
                }
                if 'page' in result:
                    row['ページ'] = result.get('page', '')
                
                # 各領域のテキストを追加
                for key, value in result.items():
                    if key not in ['filename', 'filepath', 'page', 'page_index', 'error']:
                        row[key] = value
                
                if 'error' in result:
                    row['エラー'] = result.get('error', '')
                
                display_data.append(row)
            
            if display_data:
                df = pd.DataFrame(display_data)
                st.dataframe(df, use_container_width=True)
                
                # エクスポートボタン
                st.subheader("エクスポート")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Excelエクスポート
                    if st.button("📊 Excelとしてダウンロード", type="primary"):
                        excel_data = export_to_excel(
                            st.session_state.processing_results,
                            st.session_state.selected_regions
                        )
                        st.download_button(
                            label="Excelファイルをダウンロード",
                            data=excel_data,
                            file_name="ocr_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                with col2:
                    # CSVエクスポート
                    csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📄 CSVとしてダウンロード",
                        data=csv_data,
                        file_name="ocr_results.csv",
                        mime="text/csv"
                    )


if __name__ == "__main__":
    main()

