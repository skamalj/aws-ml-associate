"""Static checks for the practice notebooks (no AWS calls are made).

Usage:  python tools/nbcheck.py [paths...]      (default: every .ipynb and common/*.py)

Checks
  1. every code cell is valid Python (IPython ! and % lines are ignored)
  2. code cells stay small (warning above MAX_LINES lines)
  3. boto3 calls: the method exists on the client, keyword arguments and nested
     dict keys exist in the API model, and required parameters are present
"""
import ast
import glob
import json
import sys
import warnings

warnings.filterwarnings("ignore")
import boto3  # noqa: E402

MAX_LINES = 30
_session = boto3.session.Session(region_name="us-east-1")
_clients = {}


def get_client(service):
    if service not in _clients:
        try:
            _clients[service] = _session.client(
                service, aws_access_key_id="x", aws_secret_access_key="x")
        except Exception:
            _clients[service] = None
    return _clients[service]


def clean(source):
    out = []
    for line in source.splitlines():
        s = line.lstrip()
        out.append("pass" if s.startswith(("!", "%")) and not line[:1].isspace()
                   else ("" if s.startswith(("!", "%")) else line))
    return "\n".join(out)


def check_value(node, shape, path, problems):
    """Recursively compare dict literals with the botocore shape."""
    if shape is None:
        return
    if isinstance(node, ast.Dict) and shape.type_name == "structure":
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        spread = any(k is None for k in node.keys)
        for k, v in zip(node.keys, node.values):
            if not isinstance(k, ast.Constant):
                continue
            if k.value not in shape.members:
                problems.append(f"unknown key '{k.value}' in {path} "
                                f"(valid: {sorted(shape.members)[:12]})")
            else:
                check_value(v, shape.members[k.value], f"{path}.{k.value}", problems)
        if not spread and len(keys) == len(node.keys):
            for req in shape.required_members:
                if req not in keys:
                    problems.append(f"missing required key '{req}' in {path}")
    elif isinstance(node, ast.List) and shape.type_name == "list":
        for el in node.elts:
            check_value(el, shape.member, path + "[]", problems)
    elif isinstance(node, ast.Dict) and shape.type_name == "map":
        for v in node.values:
            check_value(v, shape.value, path + "{}", problems)


def service_of(call):
    """Return the service name if `call` creates a boto3 client, else None."""
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return None
    if call.func.attr != "client" or not call.args:
        return None
    arg = call.args[0]
    return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None


def check_tree(tree, var2svc, problems):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt, val = node.targets[0], node.value
            pairs = [(tgt, val)]
            if isinstance(tgt, ast.Tuple) and isinstance(val, ast.Tuple):
                pairs = list(zip(tgt.elts, val.elts))
            for t, v in pairs:
                svc = service_of(v)
                if svc and isinstance(t, ast.Name):
                    var2svc[t.id] = svc
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        svc = var2svc.get(owner.id) if isinstance(owner, ast.Name) else service_of(owner)
        if not svc:
            continue
        client = get_client(svc)
        if client is None:
            problems.append(f"line {node.lineno}: unknown boto3 service '{svc}'")
            continue
        method = node.func.attr
        if not hasattr(client, method):
            problems.append(f"line {node.lineno}: {svc} client has no method '{method}'")
            continue
        op = client.meta.method_to_api_mapping.get(method)
        if not op:
            continue
        shape = client.meta.service_model.operation_model(op).input_shape
        if shape is None:
            continue
        local = []
        names = [k.arg for k in node.keywords]
        for kw in node.keywords:
            if kw.arg is None:
                continue
            if kw.arg not in shape.members:
                local.append(f"unknown parameter '{kw.arg}'")
            else:
                check_value(kw.value, shape.members[kw.arg], kw.arg, local)
        if None not in names:
            for req in shape.required_members:
                if req not in names:
                    local.append(f"missing required parameter '{req}'")
        problems += [f"line {node.lineno}: {svc}.{method}: {p}" for p in local]


def check_notebook(path):
    errors, warns, var2svc = [], [], {}
    nb = json.load(open(path))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        n = len([l for l in src.splitlines() if l.strip()])
        if n > MAX_LINES:
            warns.append(f"cell {i}: {n} lines (keep cells small)")
        first = src.lstrip().splitlines()[0] if src.strip() else ""
        if first.startswith("%%"):
            # cell magics: only %%writefile of a .py file is checked as Python
            if not (first.startswith("%%writefile") and first.strip().endswith(".py")):
                continue
            src = "\n".join(src.lstrip().splitlines()[1:])
        try:
            tree = ast.parse(clean(src))
        except SyntaxError as e:
            errors.append(f"cell {i}: syntax error: {e}")
            continue
        probs = []
        check_tree(tree, var2svc, probs)
        errors += [f"cell {i}: {p}" for p in probs]
    return errors, warns


def check_py(path):
    errors, var2svc = [], {}
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError as e:
        return [f"syntax error: {e}"], []
    check_tree(tree, var2svc, errors)
    return errors, []


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(
        glob.glob("**/*.ipynb", recursive=True) + glob.glob("common/*.py"))
    bad = 0
    for p in paths:
        errors, warns = check_notebook(p) if p.endswith(".ipynb") else check_py(p)
        status = "FAIL" if errors else "ok"
        print(f"[{status}] {p}" + (f"  ({len(warns)} warnings)" if warns else ""))
        for e in errors:
            print("    ERROR", e)
        for w in warns:
            print("    warn ", w)
        bad += bool(errors)
    sys.exit(1 if bad else 0)
