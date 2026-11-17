#!/usr/bin/env bash
# Install official Cloudflare cloudflared binary into the user's local bin.
# Source this file and call install_cloudflared_user; it does not use sudo.
set -euo pipefail

install_cloudflared_user() {
  local target_dir="${1:-$HOME/.local/bin}"
  local existing=""
  existing="$(command -v cloudflared 2>/dev/null || true)"
  if [[ -n "$existing" ]]; then
    printf '[cloudflared] existing binary: %s\n' "$existing"
    "$existing" --version
    return 0
  fi
  local arch asset dest tmp
  case "$(uname -m)" in
    x86_64|amd64) arch=amd64 ;;
    aarch64|arm64) arch=arm64 ;;
    armv7l|armv6l) arch=arm ;;
    i386|i686) arch=386 ;;
    *) printf '[cloudflared][error] unsupported Linux architecture: %s\n' "$(uname -m)" >&2; return 2 ;;
  esac
  asset="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${arch}"
  dest="$target_dir/cloudflared"
  mkdir -p "$target_dir"
  tmp="$(mktemp "$target_dir/.cloudflared.XXXXXX")"
  trap 'rm -f "$tmp"' RETURN
  printf '[cloudflared] downloading official %s binary\n' "$arch"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 20 "$asset" --output "$tmp"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --timeout=20 -O "$tmp" "$asset"
  else
    printf '[cloudflared][error] curl or wget is required\n' >&2
    return 2
  fi
  chmod 0755 "$tmp"
  "$tmp" --version
  mv -f "$tmp" "$dest"
  trap - RETURN
  PATH="$target_dir:$PATH"
  export PATH
  printf '[cloudflared] installed at %s\n' "$dest"
}
