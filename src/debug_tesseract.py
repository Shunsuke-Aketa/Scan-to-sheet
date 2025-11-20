import os
import sys
import subprocess
import platform

print("=" * 60)
print("Tesseract OCR 診断スクリプト")
print("=" * 60)

# 1. Python環境の確認
print("\n1. Python環境:")
print(f"   Python バージョン: {sys.version}")
print(f"   Python パス: {sys.executable}")
print(f"   OS: {platform.system()} {platform.version()}")

# 2. Tesseractの存在確認
print("\n2. Tesseractの存在確認:")

tesseract_paths = [
    r'C:\Users\2400125\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
]

found_path = None
for path in tesseract_paths:
    exists = os.path.exists(path)
    print(f"   {path}")
    print(f"   → {'✓ 存在する' if exists else '✗ 存在しない'}")
    if exists and not found_path:
        found_path = path
        # ファイルサイズも確認
        size = os.path.getsize(path)
        print(f"     ファイルサイズ: {size:,} bytes")

# 3. 直接実行テスト
if found_path:
    print(f"\n3. Tesseractの直接実行テスト:")
    print(f"   実行パス: {found_path}")
    
    try:
        # subprocessで直接実行
        result = subprocess.run(
            [found_path, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("   ✓ 実行成功!")
            print("   バージョン情報:")
            for line in result.stdout.split('\n')[:3]:
                if line.strip():
                    print(f"     {line}")
        else:
            print(f"   ✗ 実行失敗 (終了コード: {result.returncode})")
            if result.stderr:
                print(f"   エラー: {result.stderr}")
                
    except subprocess.TimeoutExpired:
        print("   ✗ タイムアウト")
    except Exception as e:
        print(f"   ✗ エラー: {e}")

# 4. pytesseractモジュールの確認
print("\n4. pytesseractモジュール:")
try:
    import pytesseract
    print("   ✓ pytesseractがインストール済み")
    print(f"   バージョン: {pytesseract.__version__ if hasattr(pytesseract, '__version__') else 'unknown'}")
    
    # パスを設定してテスト
    if found_path:
        print(f"\n5. pytesseractでの実行テスト:")
        pytesseract.pytesseract.tesseract_cmd = found_path
        print(f"   設定パス: {pytesseract.pytesseract.tesseract_cmd}")
        
        try:
            version = pytesseract.get_tesseract_version()
            print(f"   ✓ pytesseract経由で実行成功!")
            print(f"   バージョン: {version}")
        except pytesseract.TesseractNotFoundError as e:
            print(f"   ✗ TesseractNotFoundError: {e}")
            print("\n   【考えられる原因】")
            print("   - Tesseractの実行ファイルが破損している")
            print("   - 必要なDLLファイルが不足している")
            print("   - アンチウイルスソフトがブロックしている")
        except Exception as e:
            print(f"   ✗ エラー: {type(e).__name__}: {e}")
            
except ImportError:
    print("   ✗ pytesseractがインストールされていません")
    print("   実行: uv pip install pytesseract")

# 5. 環境変数PATHの確認
print("\n6. 環境変数PATH:")
path_env = os.environ.get('PATH', '').split(os.pathsep)
tesseract_in_path = False
for p in path_env:
    if 'tesseract' in p.lower():
        print(f"   ✓ Tesseract関連: {p}")
        tesseract_in_path = True
        
if not tesseract_in_path:
    print("   ✗ PATHにTesseractが含まれていません")

# 6. 推奨される解決策
print("\n" + "=" * 60)
print("診断結果と推奨事項:")
print("=" * 60)

if not found_path:
    print("\n【問題】Tesseractが見つかりません")
    print("\n【解決策】")
    print("1. Tesseract OCRを再インストール:")
    print("   https://github.com/UB-Mannheim/tesseract/wiki")
    print("2. インストール時に標準の場所を選択")
    
elif found_path:
    print(f"\n【状況】Tesseractは {found_path} に存在します")
    
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = found_path
        version = pytesseract.get_tesseract_version()
        print("\n✓ すべて正常です！")
        
        # 設定ファイルを作成
        with open('tesseract_config.py', 'w') as f:
            f.write(f"# Tesseract設定ファイル\n")
            f.write(f"import pytesseract\n")
            f.write(f"import os\n\n")
            f.write(f"TESSERACT_PATH = r'{found_path}'\n\n")
            f.write(f"if os.path.exists(TESSERACT_PATH):\n")
            f.write(f"    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH\n")
            f.write(f"    print(f'Tesseract設定完了: {{TESSERACT_PATH}}')\n")
        
        print("\n✓ tesseract_config.py を作成しました")
        print("  メインプログラムの最初で以下を実行してください:")
        print("  from tesseract_config import *")
        
    except Exception as e:
        print(f"\n【問題】pytesseractからTesseractを実行できません")
        print(f"エラー: {e}")
        print("\n【解決策】")
        print("1. Tesseractを管理者権限で再インストール")
        print("2. アンチウイルスソフトの除外設定に追加")
        print("3. Visual C++ 再頒布可能パッケージをインストール:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")

print("\n" + "=" * 60)
input("Enterキーを押して終了...")