#!/bin/bash
set -euo pipefail

# ============================================================
# Constants
# ============================================================
readonly CONNECT_TIMEOUT=15
readonly API_MAX_TIME=120
readonly CJ_TIMEOUT=30
readonly GIT_TIMEOUT=300
readonly PAGE_SIZE=50
readonly IMPL_MAX_LINES=200
readonly FORK_POLL_INTERVAL=2
readonly FORK_POLL_COUNT=30
readonly API_RETRY_COUNT=3
readonly API_RETRY_DELAY=2

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { printf "${GREEN}[INFO]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[NOTE]${NC} %s\n" "$*"; }
error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; exit 1; }
step()  { printf "\n${BLUE}===== %s =====${NC}\n" "$1"; }

confirm() {
    if [[ ! -t 0 ]]; then
        warn "stdin 非终端，无法确认，默认拒绝"
        return 1
    fi
    local v; read -rp "$1 (y/n): " v; [[ "$v" =~ ^[Yy]$ ]]; }

read_secret() {
    if [[ ! -t 0 ]]; then
        error "stdin 非终端，无法安全输入密码，请使用交互式终端运行"
    fi
    local prompt="$1" varname="$2" value="" char
    trap 'echo; trap - INT; return 130' INT
    printf "%s" "$prompt"
    while IFS= read -r -n1 -s char; do
        if [[ -z "$char" ]]; then
            printf '\n'; break
        elif [[ "$char" == $'\x7f' || "$char" == $'\x08' ]]; then
            if [[ -n "$value" ]]; then value="${value%?}"; printf '\b \b'; fi
        else
            value+="$char"; printf '*'
        fi
    done
    trap - INT
    printf -v "$varname" '%s' "$value"
}

format_members_md() {
    local out=""
    for m in "${MEMBERS[@]+"${MEMBERS[@]}"}"; do
        IFS='|' read -r m_name m_gcuser m_email m_role <<< "$m"
        out+="  - ${m_name}，${m_role:-成员}"$'\n'
    done
    printf '%s' "$out"
}

build_coauthor_trailers() {
    local out="" sep=""
    for m in "${MEMBERS[@]+"${MEMBERS[@]}"}"; do
        IFS='|' read -r m_name m_gcuser m_email m_role <<< "$m"
        out="${out}${sep}Co-authored-by: ${m_gcuser} <${m_email}>"
        sep=$'\n'
    done
    printf '%s' "$out"
}

build_pr_body() {
    local members_md
    members_md=$(format_members_md)
    printf '%s\n' '## 团队信息' "- 团队名称：${TEAM_SLUG}" "- 所属单位：${SCHOOL_FULL}" '- 团队成员：' "${members_md}" '## 算子信息' "- 算子名称：${OP_BASE_NAME}" "- CANNJudge 题目：${CJ_PROB_TITLE}" '' '## 算子实现介绍' "${IMPL_DESC}"
}

write_temp_json() {
    local filepath="$1" content="$2"
    printf '%s' "$content" > "$filepath"
    chmod 600 "$filepath"
}

# ============================================================
# Temp dir & cleanup
# ============================================================
TMPDIR=""
CURL_CONFIG=""
GIT_UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
GIT_CRED_FILE=""
cleanup() {
    if [[ -n "$TMPDIR" && -d "$TMPDIR" ]]; then
        rm -rf "$TMPDIR"
    fi
}
trap 'warn "正在中断，清理临时文件..."; cleanup; exit 130' INT TERM HUP
trap cleanup EXIT
TMPDIR=$(mktemp -d)
chmod 700 "$TMPDIR"
CURL_CONFIG="${TMPDIR}/.curl_config"
GIT_CRED_FILE="${TMPDIR}/.git_creds"

for cmd in jq curl python3 git; do
    command -v "$cmd" >/dev/null 2>&1 || error "缺少必要依赖: $cmd，请先安装"
done
python3 -c "import requests" 2>/dev/null || error "缺少 Python 依赖: requests，请运行 pip install requests"

CLA_SIGN_URL="https://clasign.osinfra.cn/sign-cla/68cbd4a3dbabc050b436cdd4/individual"
GC_API="https://api.gitcode.com/api/v5"

# ============================================================
# API helpers
# ============================================================
write_curl_config() {
    printf 'header = "Authorization: Bearer %s"\n' "$GC_TOKEN" > "$CURL_CONFIG"
    printf 'header = "User-Agent: %s"\n' "$GIT_UA" >> "$CURL_CONFIG"
    printf 'header = "Accept: application/json, text/plain, */*"\n' >> "$CURL_CONFIG"
    printf 'header = "Referer: https://gitcode.com/"\n' >> "$CURL_CONFIG"
    chmod 600 "$CURL_CONFIG"
}

API_CALL_SCRIPT="${TMPDIR}/api_call.sh"
cat > "$API_CALL_SCRIPT" << 'ACEOF'
#!/bin/bash
method="$1"; url="$2"; curl_config="$3"; resp_file="$4"; stderr_file="$5"
shift 5
wfmt=$'\n%{http_code}'
attempt=1; max_retry=${_API_RETRY:-3}; retry_delay=${_API_DELAY:-2}; connect_timeout=${_CONN_TIMEOUT:-15}; max_time=${_API_MAX:-120}
while true; do
    result=$(curl -sL --connect-timeout "$connect_timeout" --max-time "$max_time" -w "$wfmt" -X "$method" -K "$curl_config" "$@" "$url" 2>"$stderr_file") || true
    if [[ -n "$result" ]]; then
        http=$(printf '%s\n' "$result" | tail -1)
        if [[ "$http" =~ ^5[0-9][0-9]$ ]] && [[ "$attempt" -lt "$max_retry" ]]; then
            printf '[NOTE] 服务端错误 HTTP %s，%ds 后重试...\n' "$http" "$retry_delay" >&2
            sleep "$retry_delay"
            attempt=$((attempt + 1))
            continue
        fi
        printf '%s\n' "$result" > "$resp_file"
        break
    fi
    if [[ "$attempt" -ge "$max_retry" ]]; then
        err_content=""
        if [[ -f "$stderr_file" ]]; then err_content=$(cat "$stderr_file" 2>/dev/null); fi
        printf '[NOTE] API 请求失败: %s\n' "$err_content" >&2
        printf '\n0\n' > "$resp_file"
        rm -f "$stderr_file" 2>/dev/null || true
        exit 1
    fi
    printf '[NOTE] API 请求失败 (attempt %d/%d)，%ds 后重试...\n' "$attempt" "$max_retry" "$retry_delay" >&2
    sleep "$retry_delay"
    attempt=$((attempt + 1))
done
rm -f "$stderr_file" 2>/dev/null || true
ACEOF
chmod +x "$API_CALL_SCRIPT"

api_call() {
    local method="$1"; local url="$2"; shift 2
    local stderr_file="${TMPDIR}/curl_stderr_$$"
    local resp_file="${TMPDIR}/api_resp_$$"
    _API_RETRY=$API_RETRY_COUNT _API_DELAY=$API_RETRY_DELAY _CONN_TIMEOUT=$CONNECT_TIMEOUT _API_MAX=$API_MAX_TIME \
        bash "$API_CALL_SCRIPT" "$method" "$url" "$CURL_CONFIG" "$resp_file" "$stderr_file" "$@" || true
    cat "$resp_file"
}

http_code() {
    local code
    code=$(printf '%s\n' "$1" | tail -1)
    printf '%s' "${code:-0}"
}

http_body() {
    local body
    body=$(printf '%s\n' "$1" | sed '$d')
    printf '%s' "${body}"
}

# ============================================================
# Step 1: 输入并验证
# ============================================================
step "输入并验证信息"

# --- CANNJudge 登录 ---
while true; do
    read -rp "CANNJudge 邮箱: " CJ_EMAIL
    read_secret "CANNJudge 密码: " CJ_PASSWORD
    if [[ -z "$CJ_EMAIL" || -z "$CJ_PASSWORD" ]]; then warn "邮箱和密码不能为空"; continue; fi
    CJ_LOGIN_JSON=$(jq -n --arg email "$CJ_EMAIL" --arg password "$CJ_PASSWORD" \
        '{email: $email, password: $password}')
    CJ_LOGIN_FILE="${TMPDIR}/cj_login_body.json"
    write_temp_json "$CJ_LOGIN_FILE" "$CJ_LOGIN_JSON"
    CJ_RESP=$(curl -s --connect-timeout "$CONNECT_TIMEOUT" --max-time "$CJ_TIMEOUT" \
        -w "\n%{http_code}" -X POST -H "Content-Type: application/json" \
        -d @"$CJ_LOGIN_FILE" "https://cannjudge.cn/api/users/login" 2>/dev/null) || true
    rm -f "$CJ_LOGIN_FILE"
    CJ_HTTP=$(http_code "$CJ_RESP"); CJ_BODY=$(http_body "$CJ_RESP")
    if [[ -z "$CJ_BODY" || "$CJ_HTTP" == "0" ]]; then
        warn "网络请求失败，请检查网络连接"; continue
    fi
    if [[ "$CJ_HTTP" == "200" ]]; then
        CJ_USER_ID=$(echo "$CJ_BODY" | jq -r '._id // empty')
        CJ_NICKNAME=$(echo "$CJ_BODY" | jq -r '.nickname // "N/A"')
        if [[ -z "$CJ_USER_ID" ]]; then warn "解析用户ID失败，请重试"; continue; fi
        info "CANNJudge 登录成功! 用户: ${CJ_NICKNAME}"; break
    else
        err_msg=$(echo "$CJ_BODY" | jq -r '.message // .msg // empty' 2>/dev/null) || true
        warn "登录失败 — HTTP ${CJ_HTTP}${err_msg:+: $err_msg}，请重试"
    fi
done

# --- 算子题目地址 ---
while true; do
    read -rp "CANNJudge 题目地址 (如 https://cannjudge.cn/public/op_challenge_jiangshan_final/batch_to_space): " CJ_PROBLEM_URL
    if [[ -z "$CJ_PROBLEM_URL" ]]; then warn "不能为空"; continue; fi
    if [[ ! "$CJ_PROBLEM_URL" =~ ^https://cannjudge\.cn/ ]]; then
        warn "URL 必须以 https://cannjudge.cn/ 开头"; continue
    fi
    CJ_HOST=$(echo "$CJ_PROBLEM_URL" | sed -E 's|^https://([^/]+)/.*|\1|')
    if [[ "$CJ_HOST" != "cannjudge.cn" ]]; then
        warn "URL 域名必须为 cannjudge.cn，当前: ${CJ_HOST}"; continue
    fi
    CJ_URL_PATH=$(echo "$CJ_PROBLEM_URL" | sed 's|/*$||' | sed 's|^https://cannjudge.cn||')
    CJ_URL_PATH=$(echo "$CJ_URL_PATH" | sed 's|/submit$||')
    OP_PROBLEM_NAME=$(echo "$CJ_URL_PATH" | awk -F/ '{print $NF}')
    if [[ -z "$OP_PROBLEM_NAME" ]]; then
        warn "无法从 URL 中提取题目名"; continue
    fi
    if [[ ! "$OP_PROBLEM_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        warn "题目名包含非法字符，仅允许字母/数字/短横线/下划线"; continue
    fi
    CJ_PR_RESP=$(curl -s --connect-timeout "$CONNECT_TIMEOUT" --max-time "$CJ_TIMEOUT" \
        -w "\n%{http_code}" "https://cannjudge.cn/api/problems/name/${OP_PROBLEM_NAME}" 2>/dev/null) || true
    CJ_PR_HTTP=$(http_code "$CJ_PR_RESP"); CJ_PR_BODY=$(http_body "$CJ_PR_RESP")
    if [[ -z "$CJ_PR_BODY" || "$CJ_PR_HTTP" == "0" ]]; then
        warn "网络请求失败，请检查网络连接"; continue
    fi
    if [[ "$CJ_PR_HTTP" != "200" ]]; then
        warn "题目不存在 — HTTP ${CJ_PR_HTTP}，请检查地址"; continue
    fi
    CJ_PROBLEM_ID=$(echo "$CJ_PR_BODY" | jq -r '._id // empty')
    CJ_PROB_TITLE=$(echo "$CJ_PR_BODY" | jq -r '.title // .name')
    if [[ -z "$CJ_PROBLEM_ID" ]]; then warn "解析题目信息失败"; continue; fi
    CJ_FOUND=""; CJ_SKIP=0
    while true; do
        CJ_PAGE=$(curl -s --connect-timeout "$CONNECT_TIMEOUT" --max-time "$CJ_TIMEOUT" \
            "https://cannjudge.cn/api/submissions/user/${CJ_USER_ID}?skip=${CJ_SKIP}&limit=${PAGE_SIZE}" 2>/dev/null) || true
        CJ_PAGE_LEN=$(echo "$CJ_PAGE" | jq 'length' 2>/dev/null) || true
        if [[ "${CJ_PAGE_LEN:-0}" -eq 0 ]]; then break; fi
        CJ_FOUND=$(echo "$CJ_PAGE" | jq -r --arg pid "$CJ_PROBLEM_ID" \
            '[.[] | select(.problem._id == $pid and .status == "Pass")] | .[0] | .create_time // empty' 2>/dev/null) || true
        if [[ -n "$CJ_FOUND" ]] || [[ "$CJ_PAGE_LEN" -lt "$PAGE_SIZE" ]]; then break; fi
        CJ_SKIP=$((CJ_SKIP + PAGE_SIZE))
    done
    if [[ -z "$CJ_FOUND" ]]; then
        warn "题目存在但无 Pass 提交，请先在 CANNJudge 上通过"
        confirm "仍要继续" || continue
    else
        info "题目: ${CJ_PROB_TITLE} | 最新 Pass: $(date -d "$CJ_FOUND" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$CJ_FOUND")"
    fi
    break
done

# OP_BASE_NAME is an alias for OP_PROBLEM_NAME used in directory/branch naming
OP_BASE_NAME="$OP_PROBLEM_NAME"
info "算子名: ${OP_BASE_NAME}"

echo ""

# --- GitCode 认证 ---
while true; do
    read -rp "GitCode 用户名: " GC_USER
    echo "  访问令牌获取: GitCode → 设置 → 访问令牌 → 新建令牌 (勾选全部权限)"
    read_secret "GitCode 访问令牌: " GC_TOKEN
    if [[ -z "$GC_USER" || -z "$GC_TOKEN" ]]; then warn "用户名和令牌不能为空"; continue; fi
    write_curl_config
    GC_VR=$(api_call GET "${GC_API}/user") || true
    GC_VHTTP=$(http_code "$GC_VR"); GC_VBODY=$(http_body "$GC_VR")
    if [[ -z "$GC_VBODY" || "$GC_VHTTP" == "0" ]]; then
        warn "网络请求失败，请检查网络连接"; continue
    fi
    if [[ "$GC_VHTTP" == "200" ]]; then
        GC_LOGIN_NAME=$(echo "$GC_VBODY" | jq -r '.login // .username // empty')
        GC_EMAIL=$(echo "$GC_VBODY" | jq -r '.email // empty')
        if [[ -z "$GC_LOGIN_NAME" ]]; then warn "解析用户信息失败"; continue; fi
        info "GitCode 认证成功! 用户: ${GC_LOGIN_NAME}, 默认邮箱: ${GC_EMAIL:-未设置}"
        if [[ "$GC_LOGIN_NAME" != "$GC_USER" ]]; then
            error "令牌用户 ${GC_LOGIN_NAME} 与输入 ${GC_USER} 不一致，请使用匹配的令牌"
        fi
        break
    else
        warn "GitCode 认证失败 — HTTP ${GC_VHTTP}，请检查用户名和访问令牌"
    fi
done

echo ""

# --- 用户邮箱 ---
while true; do
    read -rp "用户邮箱 (CLA 签署邮箱): " USER_EMAIL
    if [[ -z "$USER_EMAIL" ]]; then warn "邮箱不能为空"; continue; fi
    if [[ ! "$USER_EMAIL" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
        warn "邮箱格式不正确"; continue
    fi
    break
done

# --- 邮箱一致性检查 ---
if [[ -n "${GC_EMAIL:-}" && "${GC_EMAIL}" != "${USER_EMAIL}" ]]; then
    warn "GitCode 默认邮箱 ${GC_EMAIL} 与 CLA 邮箱 ${USER_EMAIL} 不一致，通过网页创建 PR 时 CLA 校验会失败（网页操作使用默认邮箱），建议到 GitCode 设置页面修改默认邮箱为 ${USER_EMAIL}"
elif [[ -z "${GC_EMAIL:-}" ]]; then
    warn "GitCode 账号未设置默认邮箱，建议设置为 ${USER_EMAIL}"
fi

# --- CLA 签署确认 ---
echo ""
warn "竞赛仓要求所有队员签署 CLA 才能提交 PR"
echo "  签署链接: ${CLA_SIGN_URL}"
while true; do
    confirm "确认你自己 (GitCode: ${GC_LOGIN_NAME}, 邮箱: ${USER_EMAIL}) 已签署 CLA" && break
    warn "请先在浏览器中打开上述链接签署 CLA"
    echo "  签署链接: ${CLA_SIGN_URL}"
done

# --- 目标仓库地址 ---
echo ""
while true; do
    read -rp "目标仓库地址 (如 https://gitcode.com/cann/cann-ops-competitions/tree/master/03_university/CANN-ops-jiangshan-challenge-2026/preliminary/submissions): " TARGET_URL
    if [[ ! "$TARGET_URL" =~ ^https://gitcode\.com/([^/]+)/([^/]+)(/-)?/tree/([^/]+)/(.+)$ ]]; then
        warn "URL 格式不正确，应为: https://gitcode.com/{owner}/{repo}/tree/{branch}/{path}"; continue
    fi
    REPO_OWNER="${BASH_REMATCH[1]}"; REPO_NAME="${BASH_REMATCH[2]}"
    TARGET_BRANCH="${BASH_REMATCH[4]}"; TARGET_DIR="${BASH_REMATCH[5]}"
    TARGET_DIR="${TARGET_DIR%/}"
    if [[ "${REPO_OWNER}/${REPO_NAME}" != "cann/cann-ops-competitions" ]]; then
        warn "必须是竞赛仓 cann/cann-ops-competitions，当前: ${REPO_OWNER}/${REPO_NAME}"; continue
    fi
    LAST_DIR=$(basename "$TARGET_DIR")
    if [[ "$LAST_DIR" != "submissions" ]]; then
        warn "路径最后一级必须是 submissions，当前: ${LAST_DIR}"; continue
    fi
    RR=$(api_call GET "${GC_API}/repos/${REPO_OWNER}/${REPO_NAME}") || true
    RHTTP=$(http_code "$RR")
    if [[ -z "$(http_body "$RR")" || "$RHTTP" == "0" ]]; then
        warn "网络请求失败，请检查网络连接"; continue
    fi
    if [[ "$RHTTP" != "200" ]]; then
        warn "仓库不可访问 — HTTP ${RHTTP}，请检查令牌权限"; continue
    fi
    ENCODED_DIR=$(printf '%s' "$TARGET_DIR" | jq -sRr '@uri')
    DR=$(api_call GET "${GC_API}/repos/${REPO_OWNER}/${REPO_NAME}/contents/${ENCODED_DIR}?ref=${TARGET_BRANCH}") || true
    DHTTP=$(http_code "$DR")
    if [[ "$DHTTP" != "200" ]]; then
        warn "路径 ${TARGET_DIR} 在 ${TARGET_BRANCH} 分支上不存在 — HTTP ${DHTTP}，请检查 URL"; continue
    fi
    info "竞赛仓验证通过: 路径: ${TARGET_DIR}"; break
done

# --- 比赛名称 ---
echo ""
while true; do
    read -rp "比赛名称 (如 算子挑战赛江山赛区预赛): " COMPETITION_NAME
    [[ -n "$COMPETITION_NAME" ]] && break
    warn "比赛名称不能为空"
done

# --- 队伍信息 ---
echo ""
while true; do
    read -rp "学校代码缩写 (如 nju, seu, hitsz): " SCHOOL_CODE
    if [[ -z "$SCHOOL_CODE" ]]; then warn "学校代码不能为空"; continue; fi
    if [[ ! "$SCHOOL_CODE" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
        warn "只能包含字母、数字、短横线和下划线，且以字母或数字开头"; continue
    fi
    break
done
while true; do
    read -rp "队伍名称 (短横线分隔，如 op-pioneers): " TEAM_SLUG
    if [[ -z "$TEAM_SLUG" ]]; then warn "队伍名称不能为空"; continue; fi
    if [[ ! "$TEAM_SLUG" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
        warn "只能包含字母、数字、短横线和下划线，且以字母或数字开头"; continue
    fi
    break
done
TEAM_DIR_NAME="${SCHOOL_CODE}_${TEAM_SLUG}"
SUBMIT_DIR="${TARGET_DIR}/${TEAM_DIR_NAME}"
info "提交目录: ${SUBMIT_DIR}/"

# --- 团队成员信息 ---
echo ""
while true; do
    read -rp "联系人姓名: " CONTACT_NAME
    [[ -n "$CONTACT_NAME" ]] && break
    warn "联系人姓名不能为空"
done
while true; do
    read -rp "所属单位 (学校全称): " SCHOOL_FULL
    [[ -n "$SCHOOL_FULL" ]] && break
    warn "所属单位不能为空"
done

echo ""
MEMBERS=()
read -rp "你的分工 (如 算子实现): " SUBMITTER_ROLE
# Validate first member fields for pipe separator
if [[ "$CONTACT_NAME" == *"|"* || "$GC_LOGIN_NAME" == *"|"* || "$USER_EMAIL" == *"|"* || "${SUBMITTER_ROLE:-}" == *"|"* ]]; then
    error "输入不能包含 | 字符"
fi
MEMBERS+=("${CONTACT_NAME}|${GC_LOGIN_NAME}|${USER_EMAIL}|${SUBMITTER_ROLE:-负责人}")
info "已添加成员: ${CONTACT_NAME} — 你"

echo ""
info "其他团队成员"
info "  成员邮箱必须与 CLA 签署邮箱一致，否则 CLA 校验会失败"
while true; do
    read -rp "添加新成员? (y/n): " add_member
    [[ ! "$add_member" =~ ^[Yy]$ ]] && break
    while true; do
        read -rp "  姓名: " m_name
        [[ -n "$m_name" ]] && break
        warn "姓名不能为空"
    done
    while true; do
        read -rp "  GitCode 用户名: " m_gcuser
        if [[ -z "$m_gcuser" ]]; then warn "用户名不能为空"; continue; fi
        info "验证 GitCode 用户 ${m_gcuser}..."
        M_UV=$(api_call GET "${GC_API}/users/${m_gcuser}") || true
        M_UV_HTTP=$(http_code "$M_UV")
        if [[ "$M_UV_HTTP" != "200" ]]; then
            warn "GitCode 用户 ${m_gcuser} 不存在 — HTTP ${M_UV_HTTP}，请检查用户名"
            continue
        fi
        M_UV_BODY=$(http_body "$M_UV")
        m_gc_email=$(echo "$M_UV_BODY" | jq -r '.email // empty' 2>/dev/null)
        m_gc_name=$(echo "$M_UV_BODY" | jq -r '.name // .login // empty' 2>/dev/null)
        info "用户验证通过: ${m_gc_name}${m_gc_email:+, 默认邮箱: ${m_gc_email}}"
        break
    done
    while true; do
        echo "  请输入该成员签署 CLA 时使用的邮箱（必须与 GitCode 默认邮箱一致，否则 PR 将无法合入）"
        read -rp "  邮箱: " m_email
        if [[ -z "$m_email" ]]; then warn "邮箱不能为空"; continue; fi
        if [[ ! "$m_email" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
            warn "邮箱格式不正确"; continue
        fi
        if [[ -n "$m_gc_email" && "$m_gc_email" != "null" && "$m_gc_email" != "$m_email" ]]; then
            warn "GitCode 默认邮箱 ${m_gc_email} 与输入邮箱 ${m_email} 不一致，CLA 校验可能失败"
            confirm "仍要使用此邮箱" || continue
        fi
        break
    done
    read -rp "  分工 (可选): " m_role
    if [[ "$m_name" == *"|"* || "$m_gcuser" == *"|"* || "$m_email" == *"|"* || "${m_role:-}" == *"|"* ]]; then
        warn "输入不能包含 | 字符，请重新输入"
        continue
    fi
    warn "该队员是否已签署 CLA? 签署链接: ${CLA_SIGN_URL}"
    read -rp "已签署? (y/n): " mc
    if [[ ! "$mc" =~ ^[Yy]$ ]]; then
        warn "已跳过 ${m_name} — 未确认 CLA，该队员不会出现在提交中"
        continue
    fi
    MEMBERS+=("${m_name}|${m_gcuser}|${m_email}|${m_role:-成员}")
    info "已添加成员: ${m_name}"
done

# --- 算子实现介绍 ---
echo ""
if [[ -n "${EDITOR:-}" ]] && command -v "${EDITOR%% *}" >/dev/null 2>&1; then
    info "算子实现介绍 — 将打开编辑器 ${EDITOR}，保存退出即完成"
    IMPL_FILE="${TMPDIR}/impl_desc.txt"
    cat > "$IMPL_FILE" <<'EOF'
# 请在下方输入算子实现介绍，保存退出即完成
# 以 # 开头的行为注释，会被自动过滤
EOF
    ${EDITOR} "$IMPL_FILE" 2>/dev/null || warn "编辑器退出异常，使用空描述"
    IMPL_DESC=$(grep -v '^#' "$IMPL_FILE" 2>/dev/null || true)
    rm -f "$IMPL_FILE"
else
    info "算子实现介绍 — 用于 PR 描述，多行输入，单独输入 END 结束"
    IMPL_FILE="${TMPDIR}/impl_desc.txt"
    > "$IMPL_FILE"
    IMPL_LINE_COUNT=0
    while true; do
        read -rp "> " line
        [[ "$line" == "END" ]] && break
        IMPL_LINE_COUNT=$((IMPL_LINE_COUNT + 1))
        if [[ "$IMPL_LINE_COUNT" -ge "$IMPL_MAX_LINES" ]]; then
            warn "已达到最大行数 ${IMPL_MAX_LINES}，自动结束输入"
            break
        fi
        printf '%s\n' "$line" >> "$IMPL_FILE"
    done
    IMPL_DESC=$(cat "$IMPL_FILE")
    rm -f "$IMPL_FILE"
fi

# --- 提交前汇总确认 ---
echo ""
step "信息汇总"
info "算子: ${OP_BASE_NAME}"
info "比赛: ${COMPETITION_NAME}"
info "队伍: ${TEAM_DIR_NAME} — ${SCHOOL_FULL}"
info "目录: ${SUBMIT_DIR}/"
info "PR 标题: [团队提交] ${COMPETITION_NAME} ${OP_BASE_NAME} 算子提交：${TEAM_SLUG}"
info "成员数: ${#MEMBERS[@]}"
confirm "确认以上信息无误，开始提交" || error "已取消"

# ============================================================
# Step 2: 从 CANNJudge 下载算子代码
# ============================================================
step "从 CANNJudge 下载算子代码"

CJ_DIR="${TMPDIR}/cannjudge_code"
mkdir -p "$CJ_DIR"

CJ_PY_SCRIPT="${TMPDIR}/download_cj.py"
cat > "$CJ_PY_SCRIPT" << 'PYEOF'
import requests, json, os, sys
email=os.environ['CJ_EMAIL']; password=os.environ['CJ_PASSWORD']
op_name=os.environ['OP_PROBLEM_NAME']; user_id=os.environ['CJ_USER_ID']
op_id=os.environ['CJ_PROBLEM_ID']; output_dir=os.environ['CJ_DIR']
page_size=int(os.environ.get('_CJ_PAGE_SIZE','50'))
try:
    s=requests.Session()
    r=s.post('https://cannjudge.cn/api/users/login',json={'email':email,'password':password},timeout=30)
    if r.status_code!=200: print('ERROR:LOGIN_FAIL',file=sys.stderr); sys.exit(1)
    subs=[]; skip=0; ps=[]
    while not ps:
        r=s.get(f'https://cannjudge.cn/api/submissions/user/{user_id}',params={'skip':skip,'limit':page_size},timeout=30); r.raise_for_status()
        page=r.json()
        if not page: break
        ps=[x for x in page if x.get('problem',{}).get('_id','')==op_id and x['status']=='Pass']
        if not ps: skip+=page_size
    if not ps: print('ERROR:NO_PASS',file=sys.stderr); sys.exit(2)
    sid=ps[0]['_id']; print(f'最新Pass: {sid}')
    r=s.get(f'https://cannjudge.cn/api/submissions/{sid}',params={'userId':user_id},timeout=30); r.raise_for_status()
    res=r.json()
    if not res.get('can_view_code'): print('ERROR:NO_PERM',file=sys.stderr); sys.exit(3)
    saved=[]
    if 'files' in res and isinstance(res['files'],list):
        for f in res['files']:
            fp=os.path.join(output_dir,f['path'])
            real_fp=os.path.realpath(fp); real_out=os.path.realpath(output_dir)
            if not real_fp.startswith(real_out+os.sep): print(f'ERROR:PATH_TRAVERSAL:{f["path"]}',file=sys.stderr); sys.exit(8)
            os.makedirs(os.path.dirname(fp),exist_ok=True)
            with open(fp,'w') as fh: fh.write(f['content']); saved.append(f['path'])
    if not saved: print('ERROR:NO_FILES',file=sys.stderr); sys.exit(4)
    print(f'已保存{len(saved)}个文件')
except requests.exceptions.Timeout: print('ERROR:TIMEOUT',file=sys.stderr); sys.exit(5)
except requests.exceptions.ConnectionError: print('ERROR:NETWORK',file=sys.stderr); sys.exit(6)
except Exception as e: print(f'ERROR:{e}',file=sys.stderr); sys.exit(7)
PYEOF

CJ_EMAIL="$CJ_EMAIL" CJ_PASSWORD="$CJ_PASSWORD" OP_PROBLEM_NAME="$OP_PROBLEM_NAME" \
CJ_USER_ID="$CJ_USER_ID" CJ_DIR="$CJ_DIR" CJ_PROBLEM_ID="$CJ_PROBLEM_ID" \
_CJ_PAGE_SIZE=$PAGE_SIZE \
PYTHONWARNINGS=ignore python3 "$CJ_PY_SCRIPT" || error "CANNJudge 下载失败，请检查网络和账号"
unset CJ_PASSWORD  # 尽快清除，但需在 Python 脚本使用之后
rm -f "$CJ_PY_SCRIPT"

CJ_FILE_COUNT=$(find "$CJ_DIR" -type f ! -name '_meta.json' ! -name 'legacy_*' | wc -l)
CJ_FILE_COUNT=${CJ_FILE_COUNT##* }
if [[ "$CJ_FILE_COUNT" -eq 0 ]]; then
    error "下载完成但没有有效文件，请检查 CANNJudge 提交"
fi

info "下载完成 — ${CJ_FILE_COUNT} 个文件:"
find "$CJ_DIR" -type f ! -name '_meta.json' ! -name 'legacy_*' -print0 | sort -z | while IFS= read -r -d '' f; do echo "  ${f#$CJ_DIR/}"; done

# ============================================================
# Step 3: Fork 仓库
# ============================================================
step "Fork 仓库"

FORK_REPO="${GC_LOGIN_NAME}/${REPO_NAME}"
info "检查 fork: ${FORK_REPO}"
FCHK=$(api_call GET "${GC_API}/repos/${FORK_REPO}") || true
FCHK_HTTP=$(http_code "$FCHK")

if [[ "$FCHK_HTTP" == "200" ]]; then
    info "Fork 已存在: ${FORK_REPO}，将同步使用"
else
    info "创建 fork..."
    printf 'header = "Content-Type: application/json"\n' >> "$CURL_CONFIG"
    FR=$(api_call POST "${GC_API}/repos/${REPO_OWNER}/${REPO_NAME}/forks") || true
    write_curl_config
    FHTTP2=$(http_code "$FR")
    if [[ "$FHTTP2" != "200" && "$FHTTP2" != "201" && "$FHTTP2" != "202" ]]; then
        FBODY=$(http_body "$FR")
        error "Fork 失败 — HTTP ${FHTTP2}: ${FBODY}"
    fi
    info "Fork 创建成功，等待就绪..."
    for i in $(seq 1 "$FORK_POLL_COUNT"); do
        printf '\r  等待 fork 就绪... [%2ds/%ds]' "$((i * FORK_POLL_INTERVAL))" "$((FORK_POLL_COUNT * FORK_POLL_INTERVAL))"
        sleep "$FORK_POLL_INTERVAL"
        FCHK=$(api_call GET "${GC_API}/repos/${FORK_REPO}") || true
        FCHK_HTTP=$(http_code "$FCHK")
        [[ "$FCHK_HTTP" == "200" ]] && break
    done
    printf '\n'
    if [[ "$FCHK_HTTP" != "200" ]]; then
        error "Fork 仓库等待超时，请稍后手动重试"
    fi
fi

FORK_PUBLIC_URL="https://gitcode.com/${GC_LOGIN_NAME}/${REPO_NAME}.git"
UPSTREAM_URL="https://gitcode.com/${REPO_OWNER}/${REPO_NAME}.git"

# Write git credential file (avoids token in process table and .git/config)
printf 'https://oauth2:%s@gitcode.com\n' "$GC_TOKEN" > "$GIT_CRED_FILE"
chmod 600 "$GIT_CRED_FILE"

# ============================================================
# Step 4: 克隆仓库
# ============================================================
step "克隆仓库"

REPO_DIR="${TMPDIR}/repo"
info "克隆 fork..."
timeout "$GIT_TIMEOUT" git -c "credential.helper=store --file=${GIT_CRED_FILE}" \
    -c "http.userAgent=${GIT_UA}" \
    clone "$FORK_PUBLIC_URL" "$REPO_DIR" 2>&1 \
    || error "克隆失败或超时，请检查 GitCode 令牌权限和网络"

git -C "$REPO_DIR" remote add upstream "$UPSTREAM_URL" 2>/dev/null || true
info "同步上游最新代码..."
set +e
FETCH_ERR=$(timeout "$GIT_TIMEOUT" git -C "$REPO_DIR" fetch upstream "$TARGET_BRANCH" 2>&1)
FETCH_RC=$?
set -e
if [[ "$FETCH_RC" -eq 0 ]]; then
    if git -C "$REPO_DIR" rev-parse "upstream/${TARGET_BRANCH}" >/dev/null 2>&1; then
        set +e
        MERGE_ERR=$(git -C "$REPO_DIR" merge "upstream/${TARGET_BRANCH}" --ff-only 2>&1)
        MERGE_RC=$?
        set -e
        if [[ "$MERGE_RC" -ne 0 ]]; then
            warn "无法快进合并，fork 与上游有分叉"
            if confirm "尝试创建合并提交"; then
                set +e
                MERGE_ERR=$(git -C "$REPO_DIR" merge "upstream/${TARGET_BRANCH}" --no-edit 2>&1)
                MERGE_RC=$?
                set -e
            fi
        fi
        if [[ "$MERGE_RC" -ne 0 ]]; then
            warn "合并上游代码有冲突: ${MERGE_ERR}"
            if git -C "$REPO_DIR" diff --name-only --diff-filter=U 2>/dev/null | grep -q .; then
                error "存在未解决的合并冲突，请手动处理后重新运行"
            fi
        fi
    fi
else
    warn "无法 fetch upstream: ${FETCH_ERR}，继续使用 fork 当前状态"
fi

PR_BRANCH="submit/${TEAM_DIR_NAME}/${OP_BASE_NAME}"
info "创建分支: ${PR_BRANCH}"
if ! git -C "$REPO_DIR" checkout -b "$PR_BRANCH" 2>/dev/null; then
    if git -C "$REPO_DIR" checkout "$PR_BRANCH" 2>/dev/null; then
        warn "分支 ${PR_BRANCH} 已存在，重置为最新上游代码"
        RESET_BASE="upstream/${TARGET_BRANCH}"
        if ! git -C "$REPO_DIR" rev-parse "$RESET_BASE" >/dev/null 2>&1; then
            RESET_BASE="origin/${TARGET_BRANCH}"
        fi
        warn "即将执行 git reset --hard ${RESET_BASE}，这会丢弃分支上的本地修改"
        confirm "确认重置" || error "已取消，请手动处理分支后重新运行"
        git -C "$REPO_DIR" reset --hard "$RESET_BASE" 2>/dev/null \
            || error "无法重置分支到 ${RESET_BASE}"
    else
        error "无法创建或切换到分支 ${PR_BRANCH}"
    fi
fi

git -C "$REPO_DIR" config --local user.name "$GC_LOGIN_NAME"
git -C "$REPO_DIR" config --local user.email "$USER_EMAIL"

# ============================================================
# Step 5: 复制文件到提交目录
# ============================================================
step "组织提交目录"

SUBMIT_FULL_DIR="${REPO_DIR}/${SUBMIT_DIR}"
CODE_DIR="${SUBMIT_FULL_DIR}/code/${OP_BASE_NAME}"

if [[ -d "$SUBMIT_FULL_DIR" ]]; then
    warn "提交目录 ${SUBMIT_DIR}/ 已存在"
    confirm "是否覆盖" || error "请手动处理已存在的目录后重新运行"
    rm -rf "$SUBMIT_FULL_DIR"
fi

mkdir -p "$CODE_DIR"

info "目标结构: ${SUBMIT_DIR}/"
echo "  ├── README.md"
echo "  └── code/"
echo "      └── ${OP_BASE_NAME}/"

COPIED_COUNT=0
while IFS= read -r -d '' f; do
    rel="${f#$CJ_DIR/}"
    d=$(dirname "${CODE_DIR}/${rel}")
    mkdir -p "$d"
    if cp "$f" "${CODE_DIR}/${rel}" 2>/dev/null; then
        COPIED_COUNT=$((COPIED_COUNT + 1))
    else
        warn "复制失败: ${rel}"
    fi
done < <(find "$CJ_DIR" -type f ! -name '_meta.json' ! -name 'legacy_*' -print0)

if [[ "$COPIED_COUNT" -eq 0 ]]; then
    error "没有复制任何文件到提交目录"
fi

info "已复制 ${COPIED_COUNT} 个文件"
info "代码目录结构:"
cd "$CODE_DIR" && find . -type f | sort | while IFS= read -r line; do echo "  ${line}"; done; cd - >/dev/null

# ============================================================
# Step 6: 生成 README.md
# ============================================================
step "生成 README.md"

MEMBERS_MD=$(format_members_md)

{
    printf '%s\n' '## 团队信息' '' \
        "- 团队名称：${TEAM_SLUG}" \
        "- 所属单位：${SCHOOL_FULL}" \
        '- 团队成员：'
    printf '%s' "$MEMBERS_MD"
    printf '%s\n' "- 联系人：${CONTACT_NAME}" \
        "- 联系邮箱：${USER_EMAIL}" \
        '' \
        '## 算子信息' '' \
        "- 算子名称：${OP_BASE_NAME}" \
        "- CANNJudge 题目：${CJ_PROB_TITLE}"
} > "${SUBMIT_FULL_DIR}/README.md"

if [[ ! -f "${SUBMIT_FULL_DIR}/README.md" ]]; then
    error "README.md 生成失败"
fi

info "README.md 已生成"

# ============================================================
# Step 7: Commit & Push
# ============================================================
step "提交代码"

git -C "$REPO_DIR" add -A "$SUBMIT_DIR"

if git -C "$REPO_DIR" diff --cached --quiet 2>/dev/null; then
    warn "没有变更需要提交 — 目录可能已存在且内容相同"
    confirm "仍要继续推送" || error "已取消"
fi

COAUTHOR_TRAILERS=$(build_coauthor_trailers)
COMMIT_ARGS=(-m "submit ${OP_BASE_NAME} operator for ${TEAM_DIR_NAME}")
if [[ -n "$COAUTHOR_TRAILERS" ]]; then
    COMMIT_ARGS+=(-m "$COAUTHOR_TRAILERS")
fi

set +e
COMMIT_ERR=$(git -C "$REPO_DIR" commit "${COMMIT_ARGS[@]}" 2>&1)
COMMIT_RC=$?
set -e
if [[ "$COMMIT_RC" -ne 0 ]]; then
    if git -C "$REPO_DIR" diff --cached --quiet 2>/dev/null; then
        warn "没有变更需要提交"
    else
        error "git commit 失败: ${COMMIT_ERR}"
    fi
fi

info "推送到 fork..."
set +e
PUSH_OUTPUT=$(timeout "$GIT_TIMEOUT" git -c "credential.helper=store --file=${GIT_CRED_FILE}" \
    -c "http.userAgent=${GIT_UA}" \
    -C "$REPO_DIR" push --force-with-lease origin "$PR_BRANCH" 2>&1)
PUSH_RC=$?
set -e
if [[ "$PUSH_RC" -ne 0 ]]; then
    error "推送失败 — exit code ${PUSH_RC}: ${PUSH_OUTPUT}"
fi

# Clear credential file after last git operation
rm -f "$GIT_CRED_FILE"

# ============================================================
# Step 8: 创建 Pull Request
# ============================================================
step "创建 Pull Request"

EXISTING_PR_BODY=""
PR_PAGE=1
while true; do
    EXISTING_PR_RESP=$(api_call GET "${GC_API}/repos/${REPO_OWNER}/${REPO_NAME}/pulls?state=open&per_page=100&page=${PR_PAGE}") || true
    EXISTING_PR_HTTP=$(http_code "$EXISTING_PR_RESP")
    if [[ "$EXISTING_PR_HTTP" != "200" ]]; then break; fi
    PR_PAGE_BODY=$(http_body "$EXISTING_PR_RESP")
    if [[ -z "$PR_PAGE_BODY" || "$PR_PAGE_BODY" == "[]" ]]; then break; fi
    EXISTING_PR_BODY=$(echo "$EXISTING_PR_BODY" "$PR_PAGE_BODY" | jq -s 'add')
    PR_PAGE_LEN=$(echo "$PR_PAGE_BODY" | jq 'length')
    if [[ "$PR_PAGE_LEN" -lt 100 ]]; then break; fi
    PR_PAGE=$((PR_PAGE + 1))
done
if [[ -n "$EXISTING_PR_BODY" ]]; then
    EXISTING_PR_URL=$(echo "$EXISTING_PR_BODY" | jq -r --arg branch "$PR_BRANCH" --arg user "$GC_LOGIN_NAME" '[.[] | select(.head.ref == $branch and (.head.user.login // .head.user.username) == $user)] | .[0].html_url // .[0].web_url // empty' 2>/dev/null)
    if [[ -n "$EXISTING_PR_URL" ]]; then
        warn "已有 open PR: ${EXISTING_PR_URL}，将更新而非新建"
        PR_URL="$EXISTING_PR_URL"
        PR_NUMBER=$(echo "$EXISTING_PR_BODY" | jq -r --arg branch "$PR_BRANCH" --arg user "$GC_LOGIN_NAME" '[.[] | select(.head.ref == $branch and (.head.user.login // .head.user.username) == $user)] | .[0].number // .[0].iid // "unknown"' 2>/dev/null)
        info "PR #${PR_NUMBER}: ${PR_URL}"
        info "更新 PR 标题和描述..."
        PR_UPDATE_TITLE="[团队提交] ${COMPETITION_NAME} ${OP_BASE_NAME} 算子提交：${TEAM_SLUG}"
        PR_UPDATE_BODY=$(build_pr_body)
        PR_UPDATE_JSON=$(jq -n --arg title "$PR_UPDATE_TITLE" --arg body "$PR_UPDATE_BODY" \
            '{title: $title, body: $body}' 2>/dev/null) || true
        if [[ -n "$PR_UPDATE_JSON" ]]; then
            PR_UPDATE_FILE="${TMPDIR}/pr_update_body.json"
            write_temp_json "$PR_UPDATE_FILE" "$PR_UPDATE_JSON"
            printf 'header = "Content-Type: application/json"\ndata = "@%s"\n' "$PR_UPDATE_FILE" >> "$CURL_CONFIG"
            UPD_RESP=$(api_call PATCH "${GC_API}/repos/${REPO_OWNER}/${REPO_NAME}/pulls/${PR_NUMBER}") || true
            write_curl_config
            rm -f "$PR_UPDATE_FILE"
            UPD_HTTP=$(http_code "$UPD_RESP")
            if [[ "$UPD_HTTP" == "200" ]]; then
                info "PR 标题和描述已更新"
            else
                warn "PR 更新失败 — HTTP ${UPD_HTTP}，PR 内容可能为旧版本"
            fi
        fi
    fi
fi

if [[ -z "${PR_URL:-}" ]]; then

PR_TITLE="[团队提交] ${COMPETITION_NAME} ${OP_BASE_NAME} 算子提交：${TEAM_SLUG}"
PR_DESCRIPTION=$(build_pr_body)

PR_JSON=$(jq -n --arg title "$PR_TITLE" --arg body "$PR_DESCRIPTION" \
    --arg head "${GC_LOGIN_NAME}:${PR_BRANCH}" --arg base "$TARGET_BRANCH" \
    '{title: $title, body: $body, head: $head, base: $base}' 2>/dev/null) || true

if [[ -z "$PR_JSON" ]]; then
    error "构建 PR 请求体失败"
fi

PR_BODY_FILE="${TMPDIR}/pr_body.json"
write_temp_json "$PR_BODY_FILE" "$PR_JSON"
printf 'header = "Content-Type: application/json"\ndata = "@%s"\n' "$PR_BODY_FILE" >> "$CURL_CONFIG"
PR_RESP=$(api_call POST "${GC_API}/repos/${REPO_OWNER}/${REPO_NAME}/pulls") || true
write_curl_config
rm -f "$PR_BODY_FILE"
PR_HTTP=$(http_code "$PR_RESP"); PR_API_RESP=$(http_body "$PR_RESP")

if [[ -z "$PR_API_RESP" || "$PR_HTTP" == "0" ]]; then
    error "PR 创建请求失败，请检查网络连接"
fi

if [[ "$PR_HTTP" == "200" || "$PR_HTTP" == "201" ]]; then
    PR_URL=$(echo "$PR_API_RESP" | jq -r '.html_url // .web_url // .url // empty')
    PR_NUMBER=$(echo "$PR_API_RESP" | jq -r '.number // .iid // empty')
    if [[ -z "$PR_URL" || -z "$PR_NUMBER" ]]; then
        warn "PR 创建成功但解析响应失败"
        PR_URL="unknown"; PR_NUMBER="unknown"
    fi
    info "PR 创建成功! #${PR_NUMBER}: ${PR_URL}"
else
    PR_ERR_MSG=$(echo "$PR_API_RESP" | jq -r '.message // .error // .msg // empty' 2>/dev/null) || true
    if [[ "$PR_HTTP" == "400" || "$PR_HTTP" == "409" ]] && echo "$PR_ERR_MSG" | grep -qi "already exists"; then
        warn "同分支已有 PR，重新查询..."
        EXISTING_PR_URL2=$(echo "$EXISTING_PR_BODY" | jq -r --arg branch "$PR_BRANCH" --arg user "$GC_LOGIN_NAME" '[.[] | select(.head.ref == $branch and (.head.user.login // .head.user.username) == $user)] | .[0].html_url // .[0].web_url // empty' 2>/dev/null)
        EXISTING_PR_NUM2=$(echo "$EXISTING_PR_BODY" | jq -r --arg branch "$PR_BRANCH" --arg user "$GC_LOGIN_NAME" '[.[] | select(.head.ref == $branch and (.head.user.login // .head.user.username) == $user)] | .[0].number // .[0].iid // empty' 2>/dev/null)
        if [[ -n "$EXISTING_PR_URL2" ]]; then
            PR_URL="$EXISTING_PR_URL2"; PR_NUMBER="$EXISTING_PR_NUM2"
            info "已有 PR #${PR_NUMBER}: ${PR_URL}"
        else
            error "PR 创建冲突但未找到已有 PR，请手动检查"
        fi
    else
        if [[ -z "$PR_ERR_MSG" ]]; then
            PR_ERR_MSG=$(echo "$PR_API_RESP" | head -c 500)
        fi
        error "PR 创建失败 — HTTP ${PR_HTTP}${PR_ERR_MSG:+: $PR_ERR_MSG}"
    fi
fi
fi

# ============================================================
# Step 9: 触发编译流水线
# ============================================================
step "触发编译流水线"

COMPILE_BODY_FILE="${TMPDIR}/compile_body.json"
printf '{"body": "/compile"}' > "$COMPILE_BODY_FILE"
chmod 600 "$COMPILE_BODY_FILE"
printf 'header = "Content-Type: application/json"\ndata = "@%s"\n' "$COMPILE_BODY_FILE" >> "$CURL_CONFIG"
COMPILE_RESP=$(api_call POST "${GC_API}/repos/${REPO_OWNER}/${REPO_NAME}/pulls/${PR_NUMBER}/comments") || true
write_curl_config
rm -f "$COMPILE_BODY_FILE"
COMPILE_HTTP=$(http_code "$COMPILE_RESP")
if [[ "$COMPILE_HTTP" == "201" || "$COMPILE_HTTP" == "200" ]]; then
    info "已评论 /compile，编译流水线已触发"
else
    warn "评论 /compile 失败 — HTTP ${COMPILE_HTTP}，请手动在 PR 页面评论 /compile"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
info "提交完成!"
info "  算子: ${OP_BASE_NAME}"
info "  队伍: ${TEAM_DIR_NAME}"
info "  目录: ${SUBMIT_DIR}/"
info "  PR: #${PR_NUMBER} ${PR_URL}"
echo "============================================"
echo ""
warn "后续步骤: 打开 PR 页面确认 CLA 检查和编译流水线通过: ${PR_URL}"
echo ""
warn "如果 CLA 显示为 no，请确认所有团队成员的以下三者一致:"
echo "  - CLA 签署时使用的邮箱"
echo "  - GitCode 账号的默认邮箱"
echo "  - git commit 中的作者邮箱"
echo "  不一致时请修改 GitCode 默认邮箱后重新签署 CLA"
echo "  签署链接: ${CLA_SIGN_URL}"

unset GC_TOKEN
