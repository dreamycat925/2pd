# Two-Point Discrimination — Docker

この Streamlit アプリを Docker / Docker Compose でローカル実行するためのメモです。

## 通常起動

リポジトリ直下で実行します。

```bash
docker compose up -d --build
```

開く URL:

```text
http://localhost:50000
```

停止:

```bash
docker compose down
```

## 開発用起動

ソースをコンテナへマウントして、ローカル編集を反映しやすくしたい場合:

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

開く URL:

```text
http://localhost:50000
```

停止:

```bash
docker compose -f docker-compose.dev.yml down
```

## 現行のポート設定

`docker-compose.yml` / `docker-compose.dev.yml` では、以下のようにバインドされています。

```text
127.0.0.1:50000:8501
```

- コンテナ内: `8501`
- ホスト側: `50000`
- `127.0.0.1` バインドなので LAN には公開されません

## 補足

- アプリ本体の起動コマンドは `2pd_discrimination_streamlit_app.py` を前提にしています
- ポート競合がある場合は、compose ファイルの左側ポートを変更してください
  - 例: `127.0.0.1:51000:8501`
- 反映が怪しい場合は `docker compose down` の後に再度 `up -d --build` を実行してください
