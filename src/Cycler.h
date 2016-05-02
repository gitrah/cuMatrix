
class Cycler {
private:
	int idx;
	int mod;
public:
	Cycler(int idx, int mod) : idx(idx),mod(mod) {}
	Cycler(int mod) : idx(0), mod(mod) {}
	int next() { return idx++ % mod; }
};
