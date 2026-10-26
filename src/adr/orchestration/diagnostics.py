"""sequential-redacted-diagnostics-v2: redact before disk, never store raw stream.

Keep timestamps, merged stdout/stderr order, exception context and endpoint hosts.
Environment credential values, auth/cookie headers, credential assignments, URL
userinfo/query values and common key formats are removed before persistence.
"""
import json
import os
import re
from datetime import datetime,timezone
from urllib.parse import urlsplit,urlunsplit,quote
from .recovery import classify
VERSION='sequential-redacted-diagnostics-v2'

class Redactor:
    def __init__(self,environment=None):
        environment=os.environ if environment is None else environment
        secrets=set()
        for k,v in environment.items():
            if v and re.search(r'API_?KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTHORIZATION|ACCESS_TOKEN|REFRESH_TOKEN|AUTH_TOKEN|BEARER|COOKIE',k,re.I):
                secrets.update([v,quote(v,safe=''),json.dumps(v)[1:-1],repr(v)[1:-1]])
                secrets.update(v.splitlines())
        self.secrets=sorted(filter(None,secrets),key=len,reverse=True)
    def __call__(self,text):
        for value in self.secrets:text=text.replace(value,'[REDACTED]')
        # Auth/cookie header values: omit the rest of the line conservatively.
        text=re.sub(r'(?i)(\b(?:authorization|proxy-authorization|cookie|set-cookie)\b[\s\"\x27]*[:=]).*',r'\1 [REDACTED]',text)
        text=re.sub(r'(?i)\b(Bearer|Basic)\s+[A-Za-z0-9+/_.=\-]+',r'\1 [REDACTED]',text)
        key=r'(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret|secret[_-]?key|token)'
        text=re.sub(r'(?i)([\"\x27]?\b'+key+r'\b[\"\x27]?\s*[:=]\s*)(?:"[^"\n]*"|\x27[^\x27\n]*\x27|[^\s,;}]+)',r'\1[REDACTED]',text)
        text=re.sub(r'\b(?:sk-|tvly-)[A-Za-z0-9_-]+','[REDACTED]',text)
        def url(m):
            try:
                u=urlsplit(m.group(0));host=u.netloc.rsplit('@',1)[-1]
                query='&'.join(part.split('=',1)[0]+'=[REDACTED]' for part in u.query.split('&')) if u.query else ''
                return urlunsplit((u.scheme,host,u.path,query,'[REDACTED]' if u.fragment else ''))
            except ValueError:return '[REDACTED_URL]'
        return re.sub(r'https?://[^\s<>"\x27]+',url,text)

class Diagnostics:
    def __init__(self,path,environment=None):
        self.stream=path.open('x',encoding='utf-8');self.redact=Redactor(environment)
        self.blocking=set();self.categories=set();self.pending=set();self.line=0
    def record(self,line):
        safe=self.redact(line.rstrip('\n'));self.line+=1
        match=re.search(r'cleanup_id=([a-f0-9]+)',safe);identity=match.group(1) if match else 'legacy-unidentified'
        if 'BROWSER_CLEANUP_FALLBACK_SUCCEEDED:' in safe:
            kinds=['browser_cleanup_fallback_succeeded'];self.pending.discard(identity)
        elif 'BROWSER_CLEANUP_INCOMPLETE:' in safe:
            kinds=['browser_cleanup_incomplete'];self.pending.discard(identity);self.blocking.update(kinds)
        elif 'BROWSER_CLEANUP_FALLBACK:' in safe:
            kinds=['browser_cleanup_fallback_attempted'];self.pending.add(identity)
        elif 'Failed to import MCPRetriever:' in safe and 'langchain_mcp_adapters' in safe:
            kinds=['optional_mcp_import_warning']
        else:
            kinds=classify(safe);self.blocking.update(kinds)
        self.categories.update(kinds)
        self.stream.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'line':self.line,'stream':'merged_stdout_stderr','text':safe,'categories':kinds},ensure_ascii=False)+'\n');self.stream.flush()
    def close(self):
        if self.pending:self.blocking.add('browser_cleanup_outcome_unknown')
        self.stream.close()
