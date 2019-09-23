## <center>C++对象内存分析</center>

#### 1、派生类对象内存分析

```C++
class A {
public:
	virtual void func2() { cout << "A" << endl; }
	int a;

};

class B {
public:
	virtual void func2() { cout << "B" << endl; }
	int b;
};

class child : public A, public B {
public:
	virtual void func2() { cout << "C" << endl; }
	int c;
};

```

#### 2、派生对象强制转换

```C++
class A {
public:
	virtual void func2() { cout << "A" << endl; }
	int a;

};

class B {
public:
	virtual void func2() { cout << "B" << endl; }
	int b;
};

class C : public A, public B {
public:
	virtual void func2() { cout << "C" << endl; }
	int c;
};

};
int main()
{
	C* c = new C;

	A* a = new C;
	B* b = new C;

	system("pause");
	return 0;
}
```



这里分析只要分析a, b对象的位置情况。强制转换之后会将指针指向对应类地址位置。

#### 3、派生类中定义的虚函数地址存放位置

现在觉得是放在第一个父类虚函数表的后面。

