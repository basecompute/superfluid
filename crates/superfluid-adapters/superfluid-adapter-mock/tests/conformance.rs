//! The mock runtime passes the kit's conformance checks — the kit's own test that they hold for a
//! runtime that reads no artifact.

#[test]
fn the_mock_runtime_conforms() {
    superfluid_adapter_kit::conformance::runtime(&superfluid_adapter_mock::Mock);
}
