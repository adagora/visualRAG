import json
import socket

import pytest

import serve_all


def test_busy_port_fails_before_launch():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        with pytest.raises(SystemExit, match='already occupied'):
            serve_all._check_ports([listener.getsockname()[1]])


@pytest.mark.parametrize('live', [
    {'owner_token': 'someone-else', 'process_id': 123},
    {'owner_token': 'ours', 'process_id': 456},
])
def test_stop_never_signals_an_unowned_or_reused_pid(tmp_path, monkeypatch, live):
    owner = tmp_path/'services.json'
    owner.write_text(json.dumps({'owner_token':'ours','services':[{'name':'search','pid':123,'url':'http://127.0.0.1/health'}]}))
    monkeypatch.setattr(serve_all, 'OWNER_FILE', owner)
    monkeypatch.setattr(serve_all.requests, 'get', lambda *a, **k: type('Response',(),{'json':lambda self:live})())
    monkeypatch.setattr(serve_all.os, 'kill', lambda *a: pytest.fail('signaled an unrelated process'))
    serve_all.stop()


def test_stop_signals_only_matching_record(tmp_path, monkeypatch):
    owner = tmp_path/'services.json'
    owner.write_text(json.dumps({'owner_token':'ours','services':[{'name':'search','pid':123,'url':'http://127.0.0.1/health'}]}))
    monkeypatch.setattr(serve_all, 'OWNER_FILE', owner)
    monkeypatch.setattr(serve_all.requests, 'get', lambda *a, **k: type('Response',(),{'json':lambda self:{'owner_token':'ours','process_id':123}})())
    seen=[]
    monkeypatch.setattr(serve_all.os, 'kill', lambda pid, sig: seen.append(pid))
    serve_all.stop()
    assert seen == [123]
