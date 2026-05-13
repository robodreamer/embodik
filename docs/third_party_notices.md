# Third-Party Notices

EmbodiK is released under Apache-2.0. Binary wheels may bundle permissively
licensed native libraries from Pinocchio, Coal/HPP-FCL, and their dependency
stack.

The canonical notice file for release artifacts is
[`THIRD_PARTY_NOTICES.md`](https://github.com/robodreamer/embodik/blob/main/THIRD_PARTY_NOTICES.md),
with bundled license texts and native dependency inventory notices under
`third_party_licenses/`.

Before publishing repaired wheels, inspect the artifact with `auditwheel show`
on Linux or `delocate-listdeps` / `otool -L` on macOS. Confirm every bundled
native library has an accompanying license text or notice in the release
artifact.
