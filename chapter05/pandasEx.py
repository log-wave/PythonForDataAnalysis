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