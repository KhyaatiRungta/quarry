# Security Model

## Sandbox Design

The sandbox is defense in depth against a confused model, not a security boundary.

### What it does
- Subprocess isolation
- AST allowlist
- Resource limits

### Known limitations
- Network not blocked
- C extensions reachable
