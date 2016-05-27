module linalg.matrix;

void demo()
{
	StaticMatrix!(float, 2, 2) m;

	assert(m == m.identity);
	assert(-m == m * (-1));
	assert(+m == m);
	assert((m += m) == 2 * m.identity);
	assert(m + m == 2 * m);
	assert(m - m == typeof(m)(0));
	assert(m / m == m.identity);

	assert(m * m == 4 * m.identity);

	assert((m *= m) == 4 * m.identity);

	assert((m *= 2) == 8 * m.identity);

	assert(m - 6 == [
		2, -6,
		-6, 2
	].sliced(2,2));

	m[0..1, 0..1] = m[1..2, 0..1];

	assert(m == [
		0, 0, 
		0, 8
	].sliced(2,2));

	m[] = [
		7, -2,
		3, 5,
	].sliced(2,2);

	assert(m * m.inverted == m.identity);

	assert(m.transposed == m^^T);

	m = m.init;
	assert(m.vConcat(2*m) == Mat!(4,2)(
		1, 0,
		0, 1,
		2, 0,
		0, 2,
	));
	assert(m.hConcat(2*m) == Mat!(2,4)(
		1, 0, 2, 0,
		0, 1, 0, 2,
	));
}

import std.stdio;
import std.range;
import std.conv;
import std.algorithm;
import std.experimental.ndslice;
import std.experimental.logger;
import std.typecons;
import std.meta;
import std.traits;
import cblas;
import scid.linalg;
import scid.matrix;

alias Vector(uint n) = Vector!(float, n);
alias Vector(A, uint n) = StaticMatrix!(A, n, 1);

alias Matrix(uint m, uint n) = StaticMatrix!(float, m, n);

alias Vec = Vector;
alias Mat = Matrix;

Vec!(CommonType!(A, float), A.length) vec(A...)(A a)
{
	return typeof(return)(a);
}

struct StaticMatrix(A, uint m, uint n)
{
	A[m*n] data = generateIdentityMatrixData!(float, m, n);
	enum identity = StaticMatrix.init;

	enum rows = m;
	enum cols = n;

	StaticMatrix!(A,n,m) transposed() const
	{
		return typeof(return)(this[].transposed);
	}

	static if(m == n)
		StaticMatrix!(A,n,m) inverted() const
		{
			return typeof(return)(this.sliced.inverted);
		}

	auto sliced() const { return data[].sliced(m,n); }
	auto sliced(){ return data[].sliced(m,n); }
	alias sliced this;

	@disable A front();

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
	auto opUnary(string op)() const if(op == `+` || op == `-`)
	{
		auto a = cast()this;

		static if(op == `-`)
			a[] *= -1;

		return a;
	}

	auto opOpAssign(string op, B)(Slice!(2, B) that) if(op == `+` || op == `-`)
	{
		this[].opIndexOpAssign!op(that);
		return this;
	}
	auto opBinary(string op, B)(Slice!(2, B) that) const if(op == `+` || op == `-`)
	{
		auto a = cast()this;
		a.opOpAssign!op(that);

		return a;
	}

	auto opOpAssign(string op: `*`)(const(StaticMatrix) that)
	{
		this = this * that;
		return this;
	}
	auto opBinary(string op : `*`, uint k)(const(StaticMatrix!(A,n,k)) that) const
	{
		StaticMatrix!(A,m,k) result;

		this[].matrixMultiply(that[], result[]);

		return result;
	}

	auto opOpAssign(string op: `/`)(const(StaticMatrix!(A,n,n)) that) if(is(typeof(that.inverted)))
	{
		this = this / that;
		return this;
	}
	auto opBinary(string op : `/`)(const(StaticMatrix!(A,n,n)) that) const if(is(typeof(that.inverted)))
	{
		return this * that.inverted;
	}

	auto opOpAssign(string op)(A a)
	{
		this[].opIndexOpAssign!op(a);
		return this;
	}
	auto opBinary(string op)(A a) const
	{
		auto b = cast()this;
		b.opOpAssign!op(a);
		return b;
	}
	auto opBinaryRight(string op)(A a) const if(op == `*`)
	{
		return this * a;
	}

	auto opBinary(string op: `^^`)(MatrixTransposeSymbol) const
	{
		return this.transposed;
	}

	auto opEquals(const(StaticMatrix) that)
	{
		return this.data == that.data;
	}
	auto opEquals(B)(Slice!(2,B) that)
	{
		return this.sliced == that;
	}

	string toString() const
	{
		return this[].to!string;
	}
}
struct MatrixTransposeSymbol{} // idea from https://github.com/aestiff/tcbuilder/blob/master/source/app.d
enum T = MatrixTransposeSymbol();

auto matrixMultiply(A,B,C)(Slice!(2,A) a, Slice!(2,B) b, Slice!(2,C) c)
{
	assert(a.length!1 == b.length!0);

	auto m = a.length!0.to!int;
	auto n = a.length!1.to!int;
	auto k = b.length!1.to!int;

	assert(c.length!0 == m && c.length!1 == k);

	if(m == 1 && n == 1)
	{
		c[0,0] = dot(
			n, &(cast()a)[0,0], 1, &(cast()b)[0,0], 1
		);
	}
	else if(a.length!0 == 1 || a.length!1 == 1 || b.length!0 == 1 || b.length!1 == 1) 
	{
		gemv(
			CBLAS_ORDER.RowMajor,
			CBLAS_TRANSPOSE.NoTrans, 
			m, n,
			1,
			&(cast()a)[0,0], n,
			&(cast()b)[0,0], k,
			0,
			&(cast()c)[0,0], k,
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
			&(cast()a)[0,0], n,
			&(cast()b)[0,0], k,
			0,
			&(cast()c)[0,0], k,
		);
	}

	return c;
}
auto inverted(A)(Slice!(2,A) a)
in{
	assert(a.length!0 == a.length!1);
}body{
	
	alias B = Unqual!(DeepElementType!(Slice!(2,A)));

	auto workspace = new B[a.elementsCount];

	std.algorithm.copy(a.transposed.byElement, workspace);

	invert(MatrixView!(B)(
		workspace, a.length!0,
	));

	return workspace.sliced(a.length!0, a.length!1).transposed;
}

auto vConcat(Slices...)(Slices slices) if(Slices.length > 1)
in{
	foreach(s; slices)
		assert(s.length!1 == slices[0].length!1);
}body{
	auto sideLengthOf(uint i)(){ return slices[i].length!0; }
	auto byElementOf(uint i)(){ return slices[i][].byElement; }

	alias eachSlice = staticIota!(0, Slices.length);

	auto rows = staticMap!(sideLengthOf, eachSlice).only.sum;
	auto cols = slices[0].length!1;

	return staticMap!(byElementOf, eachSlice)
		.chain.array.sliced(rows, cols)
		;
}
auto hConcat(Slices...)(Slices slices) if(Slices.length > 1)
{
	alias eachSlice = staticIota!(0, Slices.length);

	auto transpose(uint i)()
	{ return slices[i].transposed; }

	return vConcat(staticMap!(transpose, eachSlice)).transposed;
}
auto dConcat(Slices...)(Slices slices) if(Slices.length > 1)
{
	alias eachSlice = staticIota!(0, Slices.length);

	auto rowsOf(uint i)(){ return slices[i].length!0; }
	auto colsOf(uint i)(){ return slices[i].length!1; }

	auto rows = staticMap!(rowsOf, eachSlice).only.sum;
	auto cols = staticMap!(colsOf, eachSlice).only.sum;

	alias E = Unqual!(CommonType!(staticMap!(DeepElementType, Slices)));

	auto workspace = (new E[rows*cols]).sliced(rows, cols);
	workspace[] = 0;

	size_t i, j;
	foreach(slice; slices)
	{
		auto i1 = i + slice.length!0;
		auto j1 = j + slice.length!1;

		workspace[i..i1, j..j1] = slice[];

		i = i1;
		j = j1;
	}

	return workspace;
}

A[m*n] generateIdentityMatrixData(A, uint m, uint n)()
{
	typeof(return) data;
	data[] = 0;

	foreach(i; staticIota!(0, min(m,n)))
		data[n*i + i] = 1;

	return data;
}
