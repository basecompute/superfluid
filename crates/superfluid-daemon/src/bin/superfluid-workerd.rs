fn main() {
    superfluid_daemon::workerd::main(std::env::args().skip(1).collect(), "superfluid-workerd")
}
