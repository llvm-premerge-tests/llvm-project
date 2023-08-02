namespace __1{

template<class T, T U>
class A;

template<class T, T U>
class A{
public:
	template<class P, P Q>
	friend class A;

	A(T x):x(x){}
	
	void foo(){}
	
private:
	T x;
};

}
