import numpy as np
import pandas as pd
from pandas import Series,DataFrame

# pandas : 표 형식의 데이터나 다양한 형태의 데이터를 다루는데 초점 (기초 : Series / DataFrame)
# numpy : 단일 산술 배열 데이터를 다루는데 특화

# ---------------------------------------------------------------------------------

# Series

# Series 는 일련의 객체를 담을 수 있는 1차원 배열 같은 자료구조이다. ( 어떠한 numpy 자료형이라도 담을 수 있다.)
# index(색인) 이라고 하는 배열의 데이터와 연관된 이름을 가지고 있다.

obj = pd.Series([4,7,-5,3])
print(obj)
print('-------------')

# Series 의 배열과 색인 객체는 각각 values 와 index 속성을 통해 얻을 수 있다.
print(obj.values)
print(obj.index) #range(4) 와 같다.
print('-------------')

# 각각의 데이터를 지칭ㅊ하는 색인을 지정하여 series객체를 생성해야 할때는 다음처럼 한다.

obj2 = pd.Series([4,7,-5,3],index=['a','b','c','d'])
print(obj2)

print(obj2['a'])
print(obj2['d'])
print(obj2[obj2>0])
print(obj2 * 2)
print('-------------')

print('b' in obj2)
print('e' in obj2)
print('-------------')

sdata = {'Ohio':35000,'Texas':71000,'Oregon':16000,'Utah':5000}
obj3 = pd.Series(sdata)
print(obj3)
print(obj3.index)
print('-------------')

states = ['California','Ohio','Oregon','Texas']
obj4 = pd.Series(sdata,index=states)
print(obj4)
print('-------------')

# sdata에 있는 값 중 3개만 확인 할 수 있는데 'California' 에 대한 값은 찾을 수 없기 때문이다.
# pandas에서 누락된 값은 NaN으로 표시된다.
# 'Utah'는 states에 포함되지 않으므로 실행결과에서 빠지게 된다.

# pandas에서 누락된 데이터를 찾을때는 isnull 과 notnull 를 이용할 수 있다.

print(pd.isnull(obj4))
print(pd.notnull(obj4))
print('-------------')

# Series 의 유용한 기능은 산술 연산에서 색인과 라벨로 자동 정렬하는 것이다.

print(obj3)
print(obj4)
print(obj3+obj4)
print('-------------')

# Series 객체와 Series 의 색인은 모두 name 속성이 있다. -> 그렇다면 values에는 name을 지정하지 못할까?

obj4.name = 'population'
obj4.index.name = 'states'
print(obj4)
print('-------------')


# Series의 색인은 대입하여 변경할 수 있다.

print(obj)
obj.index = ['Bob','Steve','Will','Lob']
print(obj)
print('-------------')

# ---------------------------------------------------------------------------------


# DataFrame

# DataFrame 은 표 같은 스프레드 시트 형식의 자료구조이고 여러 개의 컬럼이 있는데 각 컬럼은 서로 다른 종류의 값( 숫자 / 문자열 / 불리언 등) 을 담을 수 있다.
# DataFrame 은 로우와 컬럼에 대한 색인을 가지고 있는데, 색인의 모양이 같은 Series 객체를 담고 있는 파이썬 사전과 비슷하다.

# 먼저 DataFrame 객체를 같은 길이의 리스트에 담긴 사전을 이용하거나 , numpy배열을 이용해보자


data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada','Nevada'],
        'year':[200,2001,2002,2001,2002,2003],
        'pop':[1.5,1.7,3.6,2.4,2.9,3.2]}
frame = pd.DataFrame(data)
print(frame)
print('-------------')

# 큰 데이터를 다룰때는 head 메서드를 이용하여 처음 5개의 로우만 출력할 수 있다.
print(frame.head())
print('-------------')

# 원하는 순서대로 columns 를 지정하면 원하는 순서를 가진 DataFrame 객체가 생성된다.
print(pd.DataFrame(data,columns=['year','state','pop']))
print('-------------')

# series와 마찬가지로 사전에 없는 값늘 넘기면 결측치로 저장된다.

frame2 = pd.DataFrame(data,columns=['year','state','pop','debt'],
                      index=['one','two','three','four','five','six'])
print(frame2)
print('-------------')

# DataFrame의 컬럼은 Series처럼 사전 형식의 표기법으로 접근하거나 속성 형식으로 접근할 수 있다.
print(frame2['state'])
print('-------------')
print(frame2.year)
print('-------------')
print(frame2.loc['three'])
print('-------------')

# 칼럼은 대입이 가능하다. 에를들어 현재 비어 있는 'debt' 컬럼에 스칼라값이나 배열의 값을 대입할 수 있다.
frame2['debt'] = 16.5
print(frame2)
print('-------------')
frame2['debt'] = np.arange(6.)
print(frame2)
print('-------------')

# 리스트나 배열을 컬럼에 대입할 때는 대입하려는 값의 길이가 DataFrame의 크기와 동일해야 한다. series를 대입하면 DataFrame의 색인에 따라 값이 대입되며 존재하지 않는 색인에는 결측치가 대입된다.

val = pd.Series([-1.2,-1.5,-1.7],index=['two','four','five'])
print(val)
print('-------------')
frame2['debt'] = val
print(frame2)
print('-------------')
val = pd.Series([-1.5,-1.0,-0.5],index=['one','three','six'])
frame2['debt'] = val
print(frame2)
print('-------------')

# 존재하지 않는 칼럼을 대입하면 새로운 칼럼을 생성한다. 파이썬 사전형에서처럼 del 예약어를 사용하여 컬럼을 삭제할 수 있다.
# del 예약어에 대한 예제로 , state 칼럼의 값이 'Ohio'인지 아닌지에 대한 불리언값을 담고 있는 새로운 칼럼을 생성하자.

frame2['eastern'] = frame2.state == 'Ohio'
print(frame2)
print('-------------')

# del 예약어를 이용하여 'eastern'칼럼을 삭제할 수 있다.
del frame2['eastern']
print(frame2)
print('-------------')

### DataFrame 의 색인을 이용하여 얻은 컬럼은 내부 데이터에 대한 뷰(view)이며 복사가 이루어지지 않는다.
### 따라서 이렇게 얻은 Series 객체에 대한 변경은 실제 DataFrame에 반영된다.
### 복사본이 필요할때에는 Series의 copy메서드를 이용하자.


# 중첩된 사전을 이용하여 데이터를 생성할 수 있는데 이때 바깥에 있는 사전의 키는 칼럼이 되고 안에 있는 키는 로우가 된다.
pop = {'Nevada':{2001:2.4,2002:2.9},
       'Ohio':{2001:1.5,2001:1.7,2002:3.6}}
frame3 = pd.DataFrame(pop)
print(frame3)
print('-------------')

# numpy 배열과 유사한 문법으로 데이터를 전치할 수 있다.
print(frame3.T)
print('-------------')

# 중첩된 사전을 이용하여 DataFrame을 생성할 떄 안쪽에 있는 사전값은 키값별로 조합되어 결과의 색인이 되지만 색인을 직접 지정하면 지정된 색인으로 DataFrame을 생성한다.
print(pd.DataFrame(pop,index=[2001,2002,2003]))
print('-------------')

# 만일 데이터프렘의 인덱스와 칼럼에 name 속성을 지정했다면 이 역시 함께 출력된다.

frame3.index.name = 'year'
frame3.columns.name='state'
print(frame3)
print('-------------')

# values의 속성은 DataFrame에 저장된 데이터를 2차원 배열로 반환한다.
print(frame3.values)



# ---------------------------------------------------------------------------------

# 색인 객체

# pandas의 색인 객체는 표 형식의 데이터에서 각 로우와 칼럼에 대한 이름과 다른 메타데이터(축의 이름등)을 저장하는 객체이다.
# Seires나 DataFrame객체를 생성할때 사용되는 배열이나 다른 순차적인 이름은 내부적으로 색인으로 변환된다.

obj = pd.Series(range(3),index=['a','b','c'])
index = obj.index
print(index)
print('-------------')

print(index[1:])
print('-------------')

# 색인 객체는 변경이 불가능하다. -> 따라서 자료구조 사이에어 안전하게 공유될 수 있다.
labels = pd.Index(np.arange(3))
print(labels)
print('-------------')

obj2 = pd.Series([1.5,-2.5,0], index=labels)
print(obj2)

# 배열과 유사하게 index 객체도 고정 크기로 동작한다.
print(frame3)
print('-------------')
print(frame3.columns)
print('-------------')
print('Ohio ' in frame3.columns)
print(2003 in frame3.index)
print('-------------')

# 파이썬의 집합과는 달리 pandas의 인덱스는 중복되는 값을 허용한다.
dup_labels = pd.Index(['foo','foo','bar','bar'])
print(dup_labels)
print('-------------')

# ---------------------------------------------------------------------------------


# 핵심 기능
# 재색인 : pandas 객체의 중요한 기능 중 하나는 reindex인데 , 새로운 색인에 맞도도록 객체를 새로 생성한다.

obj  = pd.Series([4.5,7.2,-5.3,3.6], index=['d','b','a','c'])
print(obj)
print('-------------')
obj2 = obj.reindex(['a','b','c','d','e'])
print(obj2)
print('-------------')

# 시계열 같은 순차적인 데이터를 재색인 할때 값을 보관하거나 채워넣어야 할 경우가 있다.
# method 옵션을 이용해서 이를 해결 할 수 있으며 , ffill 같은 메서드를 이용하여 누락된 값을 직전의 값으로 채워 넣을 수 있다.

obj3 = pd.Series(['blue','purple','yellow'], index=[0,2,4])
print(obj3)
print('-------------')
print(obj3.reindex(range(6),method='ffill'))
print('-------------')

# DataFrame 에 대한 reindex는 로우,컬럼 또는 둘 다 변경이 가능하다. 그냥 순서만 전달하면 로우가 재색인된다.

frame = pd.DataFrame(np.arange(9).reshape(3,3),
                     index=['a','b','c'],
                     columns=['Ohio','Texas','California'])
print(frame)
print('-------------')
frame2 = frame.reindex(['a','b','c','d'])
print(frame2)
print('-------------')

# 재색인은 loc를 이용하여 라벨로 색인할 수 있다.


# ---------------------------------------------------------------------------------

# 하나의 로우나 컬럼을 삭제하기
# drop 메서드를 이용하여 선택한 값들이 삭제된 새로운 객체를 얻을 수 있다.

obj = pd.Series(np.arange(5.),index=['a','b','c','d','e'])
print(obj)
print('-------------')

new_obj = obj.drop('c')
print(new_obj)
print(obj)
print('-------------')

print(obj.drop(['d','c']))

# DataFrame 에서는 로우와 칼럼 모두에서 값을 삭제할 수 있다.

data = pd.DataFrame(np.arange(16).reshape(4,4),
                    index=['Ohio','Colorado','Utah','New York'],
                    columns=['one','two','three','four'])
print(data)
print('-------------')

# drop 함수에 인자로 로우 이름을 넘기면 해당 로우의 값을 모두 삭제 한다.
new_data = data.drop(['Colorado','Ohio'])
print(new_data)
print('-------------')

# 칼럼의 값을 삭제할 때는 axis=1 또는 axis = 'columns'를 인자로 넘겨주면 된다.
print(data.drop('two',axis=1))
print('-------------')
print(data.drop(['two','four'],axis='columns'))
print('-------------')

# drop 함수처럼 Seris 나 DataFrame 의 크기 또는 형태를 변경하는 함수는 새로운 객체를 반환하는 대신 원본 객체를 변경한다.

obj.drop('c',inplace=True)
print(obj)
print('-------------')

# 색인하기 , 선택하기 , 거르기

# Series의 색인은 numpy배열의 색인과 유사하게 동작하지만 정수가 아니어도 된다는 점이 다르다.

obj = pd.Series(np.arange(4.),index=['a','b','c','d'])
print(obj)
print('-------------')
print(obj['b'])
print('-------------')
print(obj[1])
print('-------------')
print(obj[['b','a','d']])
print('-------------')
print(obj[obj<2])
print('-------------')


### 라벨 이름으로 슬라이싱을하면 시작점과 끝점을 포함한다는 것이 일반 파이썬에서의 슬라이싱과 다른 점이다.

print(obj['b':'c'])
print('-------------')

obj['b':'c'] = 5
print(obj)
print('-------------')

# 색인으로 DataFrame에서 하나 이상의 칼럼 값을 가져올 수 있다.

data = pd.DataFrame(np.arange(16).reshape(4,4),
                    index=['Ohio','Colorado','Utho','New York'],
                    columns=['one','two','three','four'])
print(data)
print('-------------')
print(data['two'])
print('-------------')
print(data[['three','one']])

# 슬라이싱으로 로우를 선택하거나 불리언 배열로 로우를 선택할 수도 있다
print(data[:3])
print('-------------')
print(data[data['three']>5])
print('-------------')
# loc와 iloc로 선택하기

print(data.loc['Colorado',['one','two']])
print('-------------')

print(data.iloc[2,[3,0,1]])
print('-------------')

print(data.iloc[2])
print('-------------')


# 산술 연산과 데이터 정렬

# Pandas에서 가장 중요한 기능 중 하나는 다른 색인을 가지고 있는 객체 간의 산술 연산이다.

s1 = pd.Series([7.3,-2.5,3.4,1.5], index=['a','c','d','e'])
s2 = pd.Series([-2.1,3.6,-1.5,4,3.1],
               index=['a','c','e','f','g'])
print(s1)
print('-------------')
print(s2)
print(s1+s2)
print('-------------')


# DataFrame 과 Series간의 연산

arr = np.arange(12.).reshape((3,4))
print(arr)
print('-------------')
print(arr[0])
print('-------------')
print(arr-arr[0])
print('-------------')

# 함수 적용과 매핑

# pandas 객체에도 numpy의 유니버설 함수를 적용 할 수 있다.
frame = pd.DataFrame(np.random.randn(4,3),columns=list('bde'),
                     index=['Utho','Ohio','Texas','Oregon'])
print(frame)
print(np.abs(frame))
print('-------------')


# 정렬과 순위
obj = pd.Series(range(4),index=['d','a','b','c'])
print(obj.sort_index())

# DataFrame은 로우나 컬럼 중 하나의 축을 기준으로 정렬할 수 있다.
frame = pd.DataFrame(np.arange(8).reshape((2,4)),
                     index=['three','one'],
                     columns=['d','a','b','c'])
print(frame.sort_index)
print('-------------')

# 데이터는 기본적으로 오름차순으로 정렬되고 내림차순으로 정렬 할 수 있다.

print(frame.sort_index(axis=1,ascending=False))
print('-------------')

# Series 객체를 값에 따라 정렬하고 싶다면 sort_values 메서드를 사용하면 된다.

obj = pd.Series([4,7,-3,2])
print(obj.sort_values())
print('-------------')

# 정렬할때 비어있는 값은 기본적으로 Series객체에서 가장 마지막에 위치한다.

obj = pd.Series([4,np.nan,7,np.nan,-3,2])
print(obj.sort_values())
print('-------------')


