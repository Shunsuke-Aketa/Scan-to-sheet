# Streamlit Cloudへのデプロイ手順

このドキュメントでは、Scan To SheetアプリをStreamlit Cloudにデプロイする手順を説明します。

## 前提条件

- GitHubアカウント
- Streamlit Cloudアカウント（GitHubアカウントで無料登録可能）
- このプロジェクトがGitHubリポジトリにプッシュされていること

## デプロイ手順

### 1. GitHubリポジトリの準備

1. GitHubで新しいリポジトリを作成（または既存のリポジトリを使用）
2. ローカルプロジェクトをGitHubにプッシュ

```bash
# Gitリポジトリを初期化（まだの場合）
git init

# ファイルをステージング
git add .

# コミット
git commit -m "Initial commit: Scan To Sheet app"

# GitHubリポジトリをリモートとして追加（YOUR_USERNAMEとYOUR_REPO_NAMEを置き換え）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# メインブランチにプッシュ
git branch -M main
git push -u origin main
```

### 2. Streamlit Cloudアカウントの作成

1. [Streamlit Cloud](https://streamlit.io/cloud)にアクセス
2. "Sign up"をクリック
3. GitHubアカウントでログイン（OAuth認証）

### 3. Streamlit Cloudにアプリをデプロイ

1. Streamlit Cloudのダッシュボードにアクセス
2. "New app"ボタンをクリック
3. 以下の情報を入力：
   - **Repository**: デプロイするGitHubリポジトリを選択
   - **Branch**: `main`（またはメインブランチ名）
   - **Main file path**: `src/app.py`
   - **App URL**: カスタムURLを設定（オプション）

4. "Deploy!"ボタンをクリック

### 4. デプロイの確認

- デプロイが完了すると、Streamlit CloudがアプリのURLを生成します
- 例: `https://your-app-name.streamlit.app`
- このURLをメモしておいてください（既存サイトへの統合で使用します）

## 注意事項

### Tesseract OCRの制限

**重要**: Streamlit CloudはTesseract OCRをシステムレベルで提供していません。そのため、以下の対応が必要です：

1. **EasyOCRの使用を推奨**
   - Streamlit Cloudでは、EasyOCRがより適切に動作します
   - アプリのサイドバーで「OCRエンジン」を「EasyOCR (AI搭載・高精度)」に設定してください
   - EasyOCRは初回実行時にモデルを自動ダウンロードします

2. **Tesseractを使用する場合**
   - Streamlit CloudではTesseractが利用できないため、エラーが発生する可能性があります
   - ローカル環境でのみTesseractを使用することを推奨します

### 環境変数の設定（オプション）

Streamlit Cloudのダッシュボードで環境変数を設定できます：

1. アプリの設定画面を開く
2. "Secrets"タブを選択
3. 必要に応じて環境変数を追加

### デプロイ後の確認事項

- [ ] アプリが正常に起動するか確認
- [ ] ファイルアップロード機能が動作するか確認
- [ ] OCR機能が動作するか確認（EasyOCRを使用）
- [ ] エクスポート機能が動作するか確認

## トラブルシューティング

### デプロイエラーが発生する場合

1. **requirements.txtの確認**
   - すべての依存ライブラリが正しく記載されているか確認
   - バージョン指定が適切か確認

2. **ログの確認**
   - Streamlit Cloudのダッシュボードでログを確認
   - エラーメッセージを確認して問題を特定

3. **ファイルパスの確認**
   - `src/app.py`が正しいパスか確認
   - 相対インポートが正しく動作するか確認

### アプリが起動しない場合

- ログを確認してエラーメッセージを特定
- ローカル環境で動作確認（`streamlit run src/app.py`）
- GitHubリポジトリのファイル構造を確認

## 次のステップ

デプロイが完了したら、以下のファイルを参照して既存サイトへの統合を行ってください：

- `ocr-app.html`: 既存サイトに追加するHTMLページ
- `INTEGRATION.md`: 統合方法の詳細説明

