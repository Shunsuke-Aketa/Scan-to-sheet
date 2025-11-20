# Scan To Sheet

画像・PDFからOCRを使用してテキストを抽出し、Excel/CSVファイルに出力するWebアプリケーションです。

## 主な機能

- 📄 **画像・PDF対応**: PNG、JPEG、PDFなど複数の形式に対応
- 🎯 **ドラッグ範囲選択**: マウスでドラッグして抽出範囲を直感的に選択
- 🌐 **日本語+英語OCR**: 日本語と英語を同時に認識
- 📊 **Excel/CSV出力**: 抽出結果をExcelまたはCSV形式でダウンロード
- 🌐 **Webアプリ**: Streamlitベースで複数のPCからアクセス可能

## 必要な環境

- Python 3.8以上
- Tesseract OCR（システムに別途インストールが必要）
- 日本語OCRデータ（jpn.traineddata）

## インストール手順

### 1. uvを使用した仮想環境のセットアップ（推奨）

```bash
# uvがインストールされていない場合
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# プロジェクトディレクトリで実行
uv venv
uv pip install -r requirements.txt
```

### 2. 通常のpipを使用する場合

```bash
# 仮想環境を作成（推奨）
python -m venv venv

# 仮想環境を有効化
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# ライブラリをインストール
pip install -r requirements.txt
```

### 3. Tesseract OCRのインストール

#### Windowsの場合

1. [Tesseract OCRの公式サイト](https://github.com/UB-Mannheim/tesseract/wiki)からインストーラーをダウンロード
2. インストーラーを実行してインストール
3. デフォルトのインストールパスは以下のいずれかです：
   - `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
   - `C:\Users\<ユーザー名>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe`

#### macOSの場合

```bash
brew install tesseract
brew install tesseract-lang  # 日本語データを含む
```

#### Linuxの場合

```bash
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
sudo apt-get install tesseract-ocr-jpn  # 日本語データ
# または
sudo yum install tesseract          # CentOS/RHEL
sudo yum install tesseract-langpack-jpn  # 日本語データ
```

### 4. 日本語OCRデータの確認

日本語OCRを使用するには、`jpn.traineddata`ファイルが必要です。

#### 確認方法

```bash
tesseract --list-langs
```

`jpn`が表示されない場合は、日本語データをインストールする必要があります。

#### 日本語データのインストール（Windows）

1. [tessdataリポジトリ](https://github.com/tesseract-ocr/tessdata)から`jpn.traineddata`をダウンロード
2. Tesseractのインストールディレクトリの`tessdata`フォルダに配置
   - 例: `C:\Program Files\Tesseract-OCR\tessdata\jpn.traineddata`

#### 日本語データのインストール（macOS/Linux）

```bash
# macOS
brew install tesseract-lang

# Linux (Ubuntu/Debian)
sudo apt-get install tesseract-ocr-jpn
```

### 5. Popplerのインストール（PDF処理用、オプション）

PDFを画像に変換する機能（`pdf2image`）を使用する場合、`poppler`が必要です。ただし、**PyMuPDFがインストールされていれば、popplerなしでもPDF処理が可能です**（自動的にフォールバックされます）。

#### Windowsの場合

1. [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)から最新版をダウンロード
2. 解凍して適切な場所に配置（例: `C:\poppler`）
3. `bin`フォルダを環境変数PATHに追加
   - 例: `C:\poppler\Library\bin`をPATHに追加

または、condaを使用している場合：

```bash
conda install -c conda-forge poppler
```

#### macOSの場合

```bash
brew install poppler
```

#### Linuxの場合

```bash
sudo apt-get install poppler-utils  # Ubuntu/Debian
# または
sudo yum install poppler-utils      # CentOS/RHEL
```

**注意**: popplerがインストールされていない場合でも、PyMuPDFがインストールされていればPDF処理は動作します。アプリは自動的にPyMuPDFを使用します。

## 使用方法

### アプリケーションの起動

```bash
# uvを使用する場合
uv run streamlit run src/app.py

# 通常のpipを使用する場合
streamlit run src/app.py
```

ブラウザが自動的に開き、Webアプリケーションが表示されます。

### 基本的な使い方

1. **ファイルアップロード**
   - 「ファイルアップロード」タブで画像またはPDFファイルを選択
   - 複数のファイルを同時にアップロード可能

2. **範囲選択**
   - 「範囲選択」タブで処理するファイルを選択
   - 画像上でマウスをドラッグして抽出したい範囲を選択
   - 複数の範囲を選択可能（自動で「範囲1」「範囲2」...とナンバーが付きます）
   - PDFの場合は、全ページまたは特定のページを選択可能

3. **テキスト抽出**
   - 「テキスト抽出を実行」ボタンをクリック
   - 選択した範囲からテキストが抽出されます

4. **結果のエクスポート**
   - 「結果表示・エクスポート」タブで抽出結果を確認
   - ExcelまたはCSV形式でダウンロード可能

## 対応ファイル形式

### 画像形式
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### PDF形式
- PDF (.pdf)
  - 画像レイヤーとテキストレイヤーの両方に対応
  - 精度の高い方を自動選択

## ライブラリ一覧

- **streamlit**: Webアプリフレームワーク
- **streamlit-drawable-canvas**: ドラッグ範囲選択機能
- **opencv-python**: 画像処理
- **pytesseract**: OCR（光学文字認識）
- **pandas**: データフレーム操作とCSV/Excel出力
- **numpy**: 数値計算
- **pdf2image**: PDF→画像変換
- **PyMuPDF**: PDFテキスト抽出
- **Pillow**: 画像処理
- **openpyxl**: Excel出力

## トラブルシューティング

### Tesseractが見つからないエラー

- Tesseract OCRが正しくインストールされているか確認してください
- 環境変数PATHにTesseractのパスが追加されているか確認してください
- Windowsの場合、アプリは自動的に一般的なインストールパスを検索します

### 日本語が認識されない

- `tesseract --list-langs`で`jpn`が表示されるか確認してください
- 日本語データ（jpn.traineddata）が正しくインストールされているか確認してください
- アプリのサイドバーで「OCR言語」が「日本語+英語」に設定されているか確認してください

### PDFが読み込めない / PDF画像変換エラー

- **popplerエラーの場合**: popplerがインストールされていない場合、アプリは自動的にPyMuPDFを使用します。PyMuPDFがインストールされていれば問題ありません
- `pdf2image`と`PyMuPDF`が正しくインストールされているか確認してください
- PDFファイルが破損していないか確認してください
- 大きなPDFファイルの場合は処理に時間がかかる場合があります
- popplerをインストールしたい場合は、上記の「Popplerのインストール」セクションを参照してください

### 範囲選択ができない

- ブラウザのJavaScriptが有効になっているか確認してください
- ページをリロードしてみてください
- 別のブラウザで試してみてください

## 開発者向け情報

### プロジェクト構造

```
.
├── src/
│   ├── app.py          # Streamlitメインアプリケーション
│   ├── extractors.py   # OCR抽出機能
│   └── utils.py        # 共通ユーティリティ関数
├── requirements.txt    # 依存ライブラリ
└── README.md          # このファイル
```

### ローカル開発

```bash
# 開発モードで起動（ホットリロード有効）
streamlit run src/app.py --server.runOnSave true
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
