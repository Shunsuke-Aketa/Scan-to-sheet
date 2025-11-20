# 既存HTMLサイトへの統合手順

このドキュメントでは、既存のHTMLサイト（`https://tamron.conohawing.com/design-dx/index.html`）にOCRアプリを統合する方法を説明します。

## 統合方法の選択

以下の2つの方法から選択できます：

### 方法A: リンクページを使用（推奨）

既存サイトから`ocr-app.html`へのリンクを追加します。ユーザーはリンクをクリックすると、OCRアプリの紹介ページに移動し、そこからStreamlitアプリを開くことができます。

### 方法B: リダイレクトページを使用

既存サイトから`ocr-app-redirect.html`へのリンクを追加します。ユーザーはリンクをクリックすると、自動的にStreamlitアプリにリダイレクトされます。

## 事前準備

1. **Streamlitアプリをデプロイ**
   - `DEPLOYMENT.md`を参照してStreamlit Cloudにアプリをデプロイ
   - デプロイ後のURLをメモ（例: `https://your-app-name.streamlit.app`）

2. **HTMLファイルのURLを更新**
   - `ocr-app.html`の`YOUR_STREAMLIT_APP_URL`を実際のURLに置き換え
   - `ocr-app-redirect.html`の`YOUR_STREAMLIT_APP_URL`を実際のURLに置き換え

## 統合手順

### ステップ1: HTMLファイルをアップロード

ConoHa WINGのファイルマネージャーまたはFTPを使用して、以下のファイルを既存サイトのディレクトリにアップロードします：

- `ocr-app.html`（または`ocr-app-redirect.html`）
- 既存の`index.html`と同じディレクトリに配置することを推奨

### ステップ2: 既存のindex.htmlにリンクを追加

既存の`index.html`ファイルを編集し、OCRアプリへのリンクを追加します。

#### 例1: ナビゲーションメニューに追加

```html
<nav>
    <ul>
        <li><a href="index.html">ホーム</a></li>
        <li><a href="ocr-app.html">OCR文字認識</a></li>
        <!-- 他のメニュー項目 -->
    </ul>
</nav>
```

#### 例2: ボタンとして追加

```html
<div class="feature-section">
    <h2>便利なツール</h2>
    <a href="ocr-app.html" class="btn btn-primary">
        📄 OCR文字認識アプリ
    </a>
</div>
```

#### 例3: フッターに追加

```html
<footer>
    <div class="footer-links">
        <a href="index.html">ホーム</a>
        <a href="ocr-app.html">OCR文字認識</a>
        <!-- 他のリンク -->
    </div>
</footer>
```

### ステップ3: スタイリング（オプション）

既存サイトのデザインに合わせて、リンクのスタイルを調整してください。

```css
/* 例: ボタンスタイルのリンク */
.ocr-app-link {
    display: inline-block;
    padding: 12px 24px;
    background-color: #667eea;
    color: white;
    text-decoration: none;
    border-radius: 5px;
    transition: background-color 0.3s;
}

.ocr-app-link:hover {
    background-color: #5568d3;
}
```

## ファイル構成の例

統合後のファイル構成は以下のようになります：

```
design-dx/
├── index.html          # 既存のトップページ
├── ocr-app.html        # OCRアプリの紹介ページ（新規追加）
├── ocr-app-redirect.html  # リダイレクトページ（オプション）
├── css/
│   └── style.css
├── js/
│   └── script.js
└── images/
    └── ...
```

## 動作確認

1. **リンクの確認**
   - 既存サイトから`ocr-app.html`へのリンクが正しく動作するか確認
   - リンクをクリックしてページが表示されるか確認

2. **Streamlitアプリの確認**
   - `ocr-app.html`からStreamlitアプリへのリンクが正しく動作するか確認
   - Streamlitアプリが正常に起動するか確認

3. **リダイレクトの確認**（方法Bを使用する場合）
   - `ocr-app-redirect.html`からStreamlitアプリへの自動リダイレクトが動作するか確認

## トラブルシューティング

### リンクが404エラーになる

- ファイルが正しいディレクトリにアップロードされているか確認
- ファイル名の大文字小文字が正しいか確認（特にLinuxサーバーの場合）
- ファイルのパーミッションが適切か確認

### Streamlitアプリが開かない

- `ocr-app.html`と`ocr-app-redirect.html`のURLが正しく設定されているか確認
- Streamlit Cloudのアプリが正常にデプロイされているか確認
- ブラウザのポップアップブロッカーが有効になっていないか確認

### デザインが既存サイトと合わない

- `ocr-app.html`のCSSを既存サイトのスタイルに合わせて調整
- 既存サイトのCSSファイルを`ocr-app.html`に読み込む

## カスタマイズのヒント

### 既存サイトのスタイルに合わせる

`ocr-app.html`の`<style>`タグ内のCSSを編集して、既存サイトのデザインに合わせることができます。

### iframeで埋め込む（上級者向け）

Streamlitアプリをiframeで埋め込むことも可能です。ただし、Streamlit Cloudのセキュリティ設定により、一部の機能が制限される場合があります。

```html
<iframe 
    src="https://YOUR_STREAMLIT_APP_URL.streamlit.app" 
    width="100%" 
    height="800px"
    frameborder="0">
</iframe>
```

## 次のステップ

統合が完了したら、以下を確認してください：

- [ ] 既存サイトからOCRアプリへのリンクが正しく動作する
- [ ] Streamlitアプリが正常に起動する
- [ ] ユーザーがスムーズにアプリにアクセスできる
- [ ] モバイルデバイスでも正しく表示される

## サポート

問題が発生した場合は、以下を確認してください：

1. `DEPLOYMENT.md` - Streamlit Cloudのデプロイ手順
2. Streamlit Cloudのログ - アプリのエラー情報
3. ブラウザのコンソール - JavaScriptエラーの確認

