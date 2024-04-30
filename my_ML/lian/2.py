from datetime import datetime, timedelta

def date_now():
    today = datetime.now().date()
    num_people = int(input("请输入需要查询的人数："))
    return today, num_people

def date_input():
    num_people = int(input("请输入已接种疫苗的人数："))
    people_info = []
    for i in range(num_people):
        dose = int(input(f"请输入第{i+1}个人接种的针数（0表示未接种，1表示第一针，2表示第二针，3表示第三针）："))
        last_date_str = input(f"请输入第{i+1}个人最近一次接种的日期（格式：YYYY-MM-DD）：")
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
        people_info.append({'dose': dose, 'last_date': last_date})
    return people_info

def date_output(people_info, today):
    result = []
    for person in people_info:
        dose = person['dose']
        last_date = person['last_date']
        if dose == 0:
            next_date = today
            is_due = True
        elif dose == 1:
            next_date = last_date + timedelta(days=30)
            is_due = next_date <= today
        elif dose == 2:
            next_date = last_date + timedelta(days=180)
            is_due = next_date <= today
        else:
            next_date = ""
            is_due = False
        result.append({'is_due': is_due, 'next_date': next_date})
    return result

# 主程序
today, num_people = date_now()
people_info = date_input()
output = date_output(people_info, today)
print("接种情况如下：")
for i, person in enumerate(output):
    print(f"第{i+1}个人：", person)
