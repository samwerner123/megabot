from megabot.cli import main

from tests.test_data import fake_exchange  # noqa: F401  (фикстура)


def test_demo_backtest_writes_report(tmp_path, capsys):
    assert main(["backtest", "--demo", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "equity.png").exists()
    assert (tmp_path / "equity.csv").exists()
    assert "СИНТЕТИЧЕСКИЕ ДАННЫЕ" in capsys.readouterr().out


def test_backtest_and_sweep_on_exchange_data(fake_exchange, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    args = ["--exchange", "fakeex", "--symbols", "BTC/USDT", "ETH/USDT", "--since", "2020-01-01"]
    assert main(["fetch", *args]) == 0
    assert main(["backtest", *args, "--out", "reports"]) == 0
    assert main(["sweep", *args]) == 0
    out = capsys.readouterr().out
    assert "Держать BTC/USDT" in out
    assert "смесь 20,50,100,200" in out
    assert (tmp_path / "reports" / "equity.png").exists()


def test_too_short_history_is_a_clean_error(fake_exchange, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.chdir(tmp_path)
    code = main(["backtest", "--exchange", "fakeex", "--symbols", "BTC/USDT", "--since", "2026-10-01"])
    assert code == 1
    assert "Ошибка" in capsys.readouterr().err
