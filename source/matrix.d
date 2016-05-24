module linalg.matrix;

void demo()
{
	StaticMatrix!(float, 2, 2) m, n;

	writeln(m);
	writeln(-m);
	writeln(+m);
	writeln(m += m);
	writeln(m + m);
	writeln(m - m);
	writeln(m / m);

	writeln(m * m);

	writeln(m *= m);

	writeln(m *= 2);
	writeln(m - 6);

	m[0..1, 0..1] = m[1..2, 0..1];

	writeln(m);
}

import std.stdio;
import std.range;
import std.conv;
import std.algorithm;
import std.experimental.ndslice;
import std.experimental.logger;
import std.typecons;
import std.meta;
import cblas;

alias Vector(uint n) = Vector!(float, n);
alias Vector(A, uint n) = StaticMatrix!(A, n, 1);

alias Matrix(uint m, uint n) = StaticMatrix!(float, m, n);

alias Vec = Vector;
alias Mat = Matrix;

struct StaticMatrix(A, uint m, uint n)
{
	A[m*n] data = generateIdentityMatrixData!(float, m, n);
	enum identity = StaticMatrix.init;

	StaticMatrix!(A,n,m) transposed()
	{
		return typeof(return)(this[].transposed);
	}

	auto sliced() const { return data[].sliced(m,n); }
	auto sliced(){ return data[].sliced(m,n); }
	alias sliced this;

	this(B)(Slice!(2, B) slice)
	{
		this.sliced[] = slice;
	}
	this(Repeat!(m*n, A) as)
	{
		data[] = [as];
	}
	static if(m*n > 1) this(A a)
	{
		data[] = a;
	}

	auto opUnary(string op)() if(op == `++` || op == `--`)
	{
		this[].opIndexUnary!op;
		return this;
	}
	auto opUnary(string op)() if(op == `+` || op == `-`)
	{
		auto a = this;

		static if(op == `-`)
			a[] *= -1;

		return a;
	}

	auto opOpAssign(string op, B)(Slice!(2, B) that) if(op == `+` || op == `-` || op == `/`)
	{
		this[].opIndexOpAssign!op(that);
		return this;
	}
	auto opBinary(string op, B)(Slice!(2, B) that) if(op == `+` || op == `-` || op == `/`)
	{
		auto a = this;
		a.opOpAssign!op(that);

		static if(op == `/`)
			foreach(i, row; that.enumerate)
				foreach(j, el; row.enumerate)
					if(el == 0)
						a[i,j] = 0;

		return a;
	}

	auto opOpAssign(string op: `*`)(StaticMatrix that)
	{
		this = this * that;
		return this;
	}
	auto opBinary(string op : `*`, uint k)(StaticMatrix!(A,n,k) that)
	{
		StaticMatrix!(A,m,k) result;

		this.matrixMultiply(that, result);

		return result;
	}

	auto opOpAssign(string op)(A a)
	{
		this[].opIndexOpAssign!op(a);
		return this;
	}
	auto opBinary(string op)(A a)
	{
		auto b = this;
		b.opOpAssign!op(a);
		return b;
	}

	string toString() const
	{
		return this[].to!string;
	}
}

auto matrixMultiply(A)(Slice!(2,A*) a, Slice!(2,A*) b, Slice!(2,A*) c)
{
	assert(a.length!1 == b.length!0);

	auto m = a.length!0.to!int;
	auto n = a.length!1.to!int;
	auto k = b.length!1.to!int;

	assert(c.length!0 == m && c.length!1 == k);

	if(m == 1 && n == 1)
	{
		c[0,0] = dot(
			n, a.ptr, 1, b.ptr, 1
		);
	}
	else if(a.isVector || b.isVector) 
	{
		gemv(
			CBLAS_ORDER.RowMajor,
			CBLAS_TRANSPOSE.NoTrans, 
			m, n,
			1,
			a.ptr, n,
			b.ptr, k,
			0,
			c.ptr, k,
		);
	}
	else
	{
		gemm(
			CBLAS_ORDER.RowMajor,
			CBLAS_TRANSPOSE.NoTrans, 
			CBLAS_TRANSPOSE.NoTrans, 
			m, k, n,
			1,
			a.ptr, n,
			b.ptr, k,
			0,
			c.ptr, k,
		);
	}

	return c;
}

// traits/predicates
enum isMatrix(M) = is(M == StaticMatrix!A, A...);
bool isVector(V)(V v)
{
	return (v.length!0 == 1 || v.length!1 == 1) && !is(typeof(v.length!2));
}

// utils
auto ptr(A, uint n)(Slice!(n, A) slice)
{
	return &slice.byElement[0];
}
A[m*n] generateIdentityMatrixData(A, uint m, uint n)()
{
	typeof(return) data;
	data[] = 0;

	foreach(i; staticIota!(0, min(m,n)))
		data[n*i + i] = 1;

	return data;
}
