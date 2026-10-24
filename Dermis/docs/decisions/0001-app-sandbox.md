# ADR 0001: Developer ID builds start unsandboxed

**Status:** Accepted  
**Date:** 2026-10-22

## Context

Dermis Studio needs USB serial access, camera and microphone capture, network servers, user-selected storage, and execution of an external Python interpreter for dataset export. The Mac App Store is not an initial distribution requirement.

## Decision

Local and Developer ID builds are unsandboxed and use the hardened runtime. Signing remains optional for development. A later sandboxed flavor may be added with serial-device, camera, microphone, network client/server, and user-selected read/write entitlements.

## Consequences

Hardware and Python workflows remain straightforward. Distribution requires notarization outside the Mac App Store, and security-sensitive process launches must remain narrowly scoped and visible to the user.

