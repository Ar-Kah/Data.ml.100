import random

if __name__ == "__main__":

    doors = []
    
    for idx in range(1000):
        doors.append(["goat", "goat", "car"])
        random.shuffle(doors[idx])

    choises_ws = []
    amount_of_cars = 0
    for door in doors:
        choise = random.choice(door)
        choises_ws.append(choise)
        if choise == "car":
            amount_of_cars += 1


    prob = amount_of_cars / len(doors)
    print(f"Probability of choosing car without changing desition: {prob}")


    amount_of_cars = 0
    # change the desition
    for door in doors:
        door: list
        choice_idx = random.choice(range(len(door)))
        item_behind_door = door[choice_idx]
        if item_behind_door == "car":

            continue

        else:
            door.pop(choice_idx)
            # Monty shows another door where is the other goat
            idx_goat = 0
            for i in range(len(door)):
                if door[i] == "goat":
                    idx_goat = i
            door.pop(idx_goat)
            c = random.choice(door)

            if c == "car":
                amount_of_cars += 1

    prob = amount_of_cars / len(doors)
    print(f"Probability of choosing car changing desition: {prob}")
