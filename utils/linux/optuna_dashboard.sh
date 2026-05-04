#!/usr/bin/env bash
# ==============================================================================
# optuna_dashboard.sh — Lanza Optuna Dashboard para cualquier paquete mPES
#
# Uso:
#   ./optuna_dashboard.sh                   # Menú interactivo
#   ./optuna_dashboard.sh bayesian          # Directo a pes_ql
#   ./optuna_dashboard.sh dql               # Directo a pes_dql
#   ./optuna_dashboard.sh dqn              # Directo a pes_dqn
#   ./optuna_dashboard.sh ac               # Directo a pes_a2c
#   ./optuna_dashboard.sh transformer      # Directo a pes_trf
#   ./optuna_dashboard.sh bayesian 9090     # pes_ql en puerto 9090
#
# Requisitos:
#   - Entorno virtual activado (linux_mpes_env)
#   - optuna-dashboard instalado (pip install optuna-dashboard)
# ==============================================================================

set -euo pipefail

# ── Rutas base ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"      # mPES/ (raíz del workspace)

# ── Puerto por defecto ───────────────────────────────────────────────────────
DEFAULT_PORT=8080

# ── Paquetes y sus directorios de inputs ─────────────────────────────────────
# Orden: alias → directorio de inputs relativo al proyecto
declare -A PKG_INPUTS=(
    [pes_ql]="tabular/pes_ql/inputs"
    [pes_dql]="tabular/pes_dql/inputs"
    [pes_dqn]="ml/pes_dqn/inputs"
    [pes_a2c]="ml/pes_a2c/inputs"
    [pes_trf]="ml/pes_trf/inputs"
)

# ── Resolver alias → nombre de paquete ───────────────────────────────────────
resolve_package() {
    case "${1:-}" in
        bayesian|Bayesian|BAYESIAN|bay|1) echo "pes_ql"  ;;
        dql|DQL|ql|2)                     echo "pes_dql" ;;
        dqn|DQN|3)                        echo "pes_dqn"   ;;
        ac|a2c|actor-critic|4)            echo "pes_a2c"    ;;
        transformer|tr|5)                 echo "pes_trf"   ;;
        pes_ql|pes_dql|pes_dqn|pes_a2c|pes_trf) echo "$1" ;;
        *) return 1 ;;
    esac
}

# ── Colores para la salida en terminal ───────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

# ==============================================================================
# Funciones auxiliares
# ==============================================================================

# Imprime un mensaje de error y sale
die() {
    echo -e "${RED}❌ Error: $1${RESET}" >&2
    exit 1
}

# Busca el archivo .db más reciente en un directorio de inputs.
# Los estudios se guardan como: <date>_BAYESIAN_OPT/optuna_study_<date>.db
# Esta función encuentra el más reciente por orden alfabético (YYYY-MM-DD).
find_latest_db() {
    local inputs_dir="$1"
    local db_path

    # Buscar archivos .db dentro de subdirectorios *_BAYESIAN_OPT/
    # Ordenar por nombre (las fechas YYYY-MM-DD se ordenan alfabéticamente)
    # y tomar el último (más reciente)
    db_path=$(find "${inputs_dir}" -path "*_BAYESIAN_OPT/optuna_study_*.db" -type f 2>/dev/null \
              | sort | tail -n 1)

    echo "${db_path}"
}

# Verifica que optuna-dashboard esté disponible
check_dependencies() {
    if ! command -v optuna-dashboard &>/dev/null; then
        die "optuna-dashboard no encontrado.\n   Instálalo con: pip install optuna-dashboard"
    fi
}

# Lanza el dashboard para un archivo .db dado
launch_dashboard() {
    local db_path="$1"
    local port="${2:-${DEFAULT_PORT}}"

    # Verificar que el archivo existe
    [[ -f "${db_path}" ]] || die "No se encontró la base de datos: ${db_path}"

    # Construir la URI de SQLite (ruta absoluta con triple /)
    local abs_path
    abs_path="$(cd "$(dirname "${db_path}")" && pwd)/$(basename "${db_path}")"
    local sqlite_uri="sqlite:///${abs_path}"

    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
    echo -e "  ${GREEN}Optuna Dashboard${RESET}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
    echo -e "  ${BLUE}Base de datos:${RESET}  ${db_path}"
    echo -e "  ${BLUE}Puerto:${RESET}         ${port}"
    echo -e "  ${BLUE}URL:${RESET}            ${GREEN}http://localhost:${port}${RESET}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
    echo -e "  Presiona ${YELLOW}Ctrl+C${RESET} para detener el servidor."
    echo ""

    # Lanzar optuna-dashboard
    optuna-dashboard "${sqlite_uri}" --port "${port}"
}

# ==============================================================================
# Menú interactivo
# ==============================================================================
show_menu() {
    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
    echo -e "  ${GREEN}Optuna Dashboard Launcher${RESET}"
    echo -e "${BOLD}════════════════════════════════════════════════════════════${RESET}"
    echo ""

    # Detectar estudios disponibles para cada paquete
    local idx=1
    declare -A MENU_DB=()
    declare -A MENU_PKG=()
    for pkg in pes_ql pes_dql pes_dqn pes_a2c pes_trf; do
        local db
        db=$(find_latest_db "${PROJECT_DIR}/${PKG_INPUTS[$pkg]}")
        MENU_PKG[$idx]="$pkg"
        MENU_DB[$idx]="$db"
        if [[ -n "${db}" ]]; then
            echo -e "  ${GREEN}${idx})${RESET} ${pkg}   ${BLUE}→${RESET} $(basename "${db}")"
        else
            echo -e "  ${RED}${idx})${RESET} ${pkg}   ${RED}(sin estudios)${RESET}"
        fi
        idx=$((idx + 1))
    done

    echo -e "  ${YELLOW}q)${RESET} Salir"
    echo ""
    read -rp "  Selección [1-5/q]: " choice

    case "${choice}" in
        [1-5])
            local sel_db="${MENU_DB[$choice]}"
            local sel_pkg="${MENU_PKG[$choice]}"
            [[ -n "${sel_db}" ]] || die "No se encontró ningún estudio en ${PKG_INPUTS[$sel_pkg]}"
            launch_dashboard "${sel_db}"
            ;;
        q|Q)
            echo -e "  ${BLUE}Hasta luego.${RESET}"
            exit 0
            ;;
        *)
            die "Opción inválida: ${choice}"
            ;;
    esac
}

# ==============================================================================
# Punto de entrada
# ==============================================================================

# Verificar dependencias
check_dependencies

# Procesar argumentos
PROJECT="${1:-}"
PORT="${2:-${DEFAULT_PORT}}"

if [[ -z "${PROJECT}" ]]; then
    # Sin argumentos: menú interactivo
    show_menu
else
    # Argumento directo: resolver paquete
    PKG_NAME="$(resolve_package "$PROJECT")" || {
        die "Proyecto desconocido: '${PROJECT}'\n   Uso: $0 [bayesian|dql|dqn|ac|transformer] [puerto]"
    }
    db=$(find_latest_db "${PROJECT_DIR}/${PKG_INPUTS[$PKG_NAME]}")
    [[ -n "${db}" ]] || die "No se encontró ningún estudio en ${PKG_INPUTS[$PKG_NAME]}"
    launch_dashboard "${db}" "${PORT}"
fi
