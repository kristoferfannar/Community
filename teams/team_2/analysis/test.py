for size in range(2, 15, 2):
    SIZE = size
    members = list(range(SIZE))

    def select(start=0, partnered=[False] * SIZE):
        # found one partnering
        if all(partnered):
            return 1

        selections = 0
        for mid in range(start, len(members)):
            if partnered[mid]:
                continue
            partnered[mid] = True
            for m2 in range(mid + 1, len(members)):

                if partnered[m2]:
                    continue

                partnered[m2] = True
                next = m2 + 1 if m2 == mid + 1 else mid + 1
                selections += select(next, partnered.copy())
                partnered[m2] = False

            partnered[mid] = False

        return selections

    print(f"size={size}: {select()}")
