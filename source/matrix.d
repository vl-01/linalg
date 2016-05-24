module linalg.matrix;

@("DEMO") unittest
{
	auto m1 = Mat!(3,3)(
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	);
	assert(m1 * Mat!(3,3).identity == m1);

	auto m2 = Mat!(2,3)(m1[1..3, 0..3]);
	assert(m2 == Mat!(2,3)(
		4, 5, 6,
		7, 8, 9,
	));

	assert(m2 * m1 == Mat!(2,3)(
		66, 81, 96,
		102, 126, 150,
	));

	m1[0..2, 0..2] = Mat!(2,2).identity;
	assert(m1 == Mat!(3,3)(
		1, 0, 3,
		0, 1, 6,
		7, 8, 9,
	));

	auto v = Vector!(float, 2).e!1;
	assert(m1[0..2, 0..2] * v == v);

	auto u = Vec!2(7,8);

	//assert(u + v == Vec!2(7,9)); TODO matrix/vector addition

	auto dot = v.transposed*u;
	assert(dot[].shape == [1,1]);
	assert(dot[0,0] == 8);

	auto outer = v * u.transposed;
	assert(outer == Mat!(2,2)(
		0, 0,
		7, 0,
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
import cblas;

alias Vector(uint n) = Vector!(float, n);
alias Vector(A, uint n) = Matrix!(A, n, 1);

alias Matrix(uint m, uint n) = Matrix!(float, m, n);
alias SubMatrix(uint m, uint n) = SubMatrix!(float, m, n);

alias Vec = Vector;
alias Mat = Matrix;

struct Matrix(A, uint m, uint n)
{
	alias Sub = SubMatrix!(A, m, n);

	A[m*n] data = [Repeat!(m*n, 0)];

	static Matrix identity()
	{
		return Matrix(generateIdentityMatrixData!(A,m,n));
	}
	Matrix!(A,n,m) transposed()
	{
		return typeof(return)(this[].transposed);
	}

	static if(m*n > 1) this(A a)
	{
		data[] = a;
	}
	this(Repeat!(m*n, A) as)
	{
		data = [as];
	}
	this(A[m*n] as)
	{
		data = as;
	}
	this(Slice!(2, A[]) slice)
	{
		this = (this[] = slice).base;
	}
	this(Matrix matrix)
	{
		this.data = matrix.data;
	}
	this(uint k, uint p)(SubMatrix!(A,k,p) subMatrix)
	{
		this = (this[] = subMatrix).base;
	}

	Matrix opOpAssign(string op, B)(B that)
	{
		static if(isMatrix!B)
			static assert(is(B == Matrix) && m == n,
				"cannot " ~ Matrix.stringof ~ " *= " ~ B.stringof
			);

		return this = this.opBinary!op(that);
	}
	Matrix!(A, m, k) opBinary(string op : "*", uint k)(Matrix!(A, n, k) that)
	{
		return typeof(return)(this[] * that[]);
	}
	Matrix opBinary(string op, B)(B that)
	{
		return Matrix(Sub(this).opBinary!op(that));
	}
	Matrix opUnary(string op)()
	{
		return Matrix(Sub(this).opUnary!op);
	}

	auto ref opIndexOpAssign(string op, B, I...)(B that, I indices)
	{
		auto binary()(){ return mixin(q{ this }~op~q{ that }); }
		auto access()(){ return mixin(q{ this.opIndex(indices) } ~op~ q{ that }); }

		auto assignedValue()
		{
			static if(is(typeof(access())))
				return access;
			else
				return binary;
		}

		return this.opIndexAssign(assignedValue, indices);
	}
	auto ref opIndexAssign(B, I...)(B that, I indices)
	{
		auto ref assign()()
		{
			static if(isMatrix!B)
			{
				auto saved = that[];
				auto item = saved.slice;
			}
			else
				auto item = that;

			return this.data[].arraySlice(m,n)
				.opIndexAssign(item, indices)
				;
		}

		static if(is(typeof(assign()) == void))
		{
			assign;
			return Sub(this).opIndex(indices);
		}
		else 
		{
			return assign;
		}
	}
	auto ref opIndex(B...)(B args)
	{
		return Sub(this).opIndex(args);
	}
	auto opSlice(uint d, B...)(B args)
	{
		return Sub(this).opSlice!d(args);
	}
	bool opEquals(B...)(B args)
	{
		return this[] == args;
	}

	alias e = basis;
	static Matrix basis(uint i)() if((m == 1 || n == 1) && i < max(m,n))
	{
		Matrix mat;
		mat.data[i] = 1;
		return mat;
	}

	string toString()
	{
		return Sub(this).toString;
	}
}
struct SubMatrix(A, uint m, uint n)
{
	private this(Matrix!(A, m, n) base)
	{
		this.base = base;
	}
	private this(Slice!(2, A[]) slice)
	{
		rows = slice.length!0;
		cols = slice.length!1;

		this.slice[] = slice;
	}
	@disable this();

	size_t length(uint i = 0)(){ return slice.length!i; }

	SubMatrix!(A, n, m) transposed()
	{
		return typeof(return)(this.slice.transposed);
	}

	template opDispatch(string op)
	{
		auto ref opDispatch(B...)(B args)
		{
			auto ref ans()(){ return mixin(q{slice.}~op~q{(args)}); }

			static if(is(typeof(ans()) == Slice!(2, A[])))
				return SubMatrix(ans);
			else static if(is(typeof(ans()) == void))
				return this;
			else
				return ans;
		}
	}

	alias opIndex = opDispatch!"opIndex";

	auto opSlice(uint d, B...)(B args)
	{
		return slice.opSlice!d(args);
	}

	auto opAssign(Slice!(2, A[]) slice)
	{
		this.slice[] = slice;
		return this;
	}
	auto opAssign(uint k, uint p)(SubMatrix!(A, k, p) subMatrix)
	{
		return this = subMatrix.slice;
	}
	auto opAssign(uint k, uint p)(Matrix!(A, k, p) matrix)
	{
		return this = matrix[].slice;
	}

	auto opOpAssign(string op)(A a) if(only("*", "/").canFind(op))
	{
		foreach(ref b; this.slice.byElement)
			mixin(q{b }~op~q{= a;});

		return this;
	}
	auto opOpAssign(string op, B)(Slice!(2, B[]) slice)
	{
		foreach(ref a, b; this.slice.byElement.lockstep(slice.byElement))
			mixin(q{a }~op~q{= b;});

		return this;
	}
	auto opOpAssign(string op : "*", uint k, uint p)(SubMatrix!(A, k, p) that)
	{
		auto ans = this * that;
		this.rows = ans.rows;
		this.cols = ans.cols;
		return this = ans;
	}
	auto opOpAssign(string op : "+", uint k, uint p)(SubMatrix!(A, k, p) that)
	{
		return this.slice[] += that.slice[];
	}
	auto opOpAssign(string op, uint k, uint p)(Matrix!(A, k, p) that)
	{
		static if(op == "*")
			this *= that[];
		else static if(op == "+")
			this.slice[] += that[].slice;
		else static assert(0);

		return this;
	}

	auto opBinary(string op : "*", uint k, uint p)(SubMatrix!(A, k, p) that)
	{
		assert(this.cols == that.rows);

		auto inner = this.cols.to!int;

		auto rows = this.rows.to!int;
		auto cols = that.cols.to!int;

		Matrix!(A,m,p) mat;

		if(rows == 1 && cols == 1)
		{
			mat[0,0] = dot(
				inner, this.slice.ptr, 1, that.slice.ptr, 1
			);
		}
		else if(this.isVector || that.isVector) // REVIEW
		{
			gemv(
				CBLAS_ORDER.RowMajor,
				CBLAS_TRANSPOSE.NoTrans, 
				rows, inner,
				1,
				this.slice.ptr, n,
				that.slice.ptr, p,
				0,
				mat.data.ptr, p,
			);
		}
		else
		{
			gemm(
				CBLAS_ORDER.RowMajor,
				CBLAS_TRANSPOSE.NoTrans, 
				CBLAS_TRANSPOSE.NoTrans, 
				rows, cols, inner,
				1,
				this.slice.ptr, n,
				that.slice.ptr, p,
				0,
				mat.data.ptr, p,
			);
		}

		return mat[0..rows, 0..cols];
	}
	auto opBinary(string op, B...)(B args)
	{
		auto copy = this;
		return copy.opOpAssign!op(args);
	}
	auto opUnary(string op)()
	{
		auto copy = this;

		foreach(ref a; copy.slice.byElement)
			a = mixin(op~q{a});

		return copy;
	}

	bool opEquals(B)(Slice!(2, B[]) that)
	{
		return this.slice == that;
	}
	bool opEquals(B)(B that) if(!is(B == Slice!C, C...))
	{
		return this == that[].slice;
	}

	string toString()
	{
		return slice.text;
	}

	private
	{
		Slice!(2, A[]) slice()
		{
			return base.data[].arraySlice(m,n)[0..rows, 0..cols];
		}

		Matrix!(A,m,n) base;
		size_t rows = m, cols = n;
	}
}

// traits/predicates
enum isMatrix(M) = is(M == Matrix!A, A...) || is(M == SubMatrix!A, A...);
bool isVector(V)(V v)
{
	return (v.length!0 == 1 || v.length!1 == 1) && !is(typeof(v.length!2));
}

// utils
size_t volume(S)(S space)
{
	return space.shape[].reduce!((a,b) => a*b);
}
auto arraySlice(A...)(A a)
{
	return std.experimental.ndslice.sliced!(Flag!"replaceArrayWithPointer".no)(a);
}
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

unittest
{
	auto x = Matrix!(float, 2, 2).identity + [ 0, 1, 0, 0 ].arraySlice(2,2);

	auto y = x[];

	assert(y * y == Matrix!(2,2)(
		1, 2, 
		0, 1
	));

	assert(x * x * x == Mat!(2,2)(
		1, 3,
		0, 1
	));

	// because arbitrary slices can't carry static dimension information for storage,
	// operations with them are elementwise
	// this might change later with dynamic array based matrices
	assert(
		x * [
			6, 7,
			8, 1
		].arraySlice(2,2)
		== Mat!(2,2)(
			6, 7,
			0, 1
		)
	);

	assert(-x == Mat!(2,2)(
		-1, -1,
		-0, -1
	));

	auto z = Matrix!(3,2).identity;

	assert(z == Mat!(3,2)(
		1, 0,
		0, 1,
		0, 0,
	));
	assert(z.transposed == Mat!(2,3)(
		1, 0, 0,
		0, 1, 0,
	));

	auto q = Vector!2.e!1;

	// TODO can't transpose vectors yet
	//assert(
	//	z * q == z[1, 0..2].transposed
	//);
}
unittest
{
	auto x = Matrix!(5,5).identity;

	auto y = x[2..5, 2..5] = Matrix!(3,3).identity * 2;

	assert(y * Vec!3(2,3,4) == Vec!3(4,6,8));

	assert(
		Mat!(6,5)(
			 0,  0,  0, 0,0,
			 0,  0,  0, 0,0,
			 1,  2,  3, 0,0,
			 4,  5,  6, 0,0,
			 7,  8,  9, 0,0,
			10, 11, 12, 0,0,
		)[3..6, 0..3]
		* Mat!(3,3).identity
		== Mat!(3,3)(
			 4,  5,  6,
			 7,  8,  9,
			10, 11, 12,
		)
	);
	assert(
		Mat!(6,5)(
			 0,  0,  0, 0,0,
			 0,  0,  0, 0,0,
			 1,  2,  3, 0,0,
			 4,  5,  6, 0,0,
			 7,  8,  9, 0,0,
			10, 11, 12, 0,0,
		)[2..6, 0..3] * Vec!3(-2,1,0)
		== Vec!4(0,-3,-6,-9)
	);
}
