# The authors of this work have released all rights to it and placed it
# in the public domain under the Creative Commons CC0 1.0 waiver
# (http://creativecommons.org/publicdomain/zero/1.0/).
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Retrieved from: http://en.literateprograms.org/Bernoulli_numbers_(Python)?oldid=5312


from gmpy import mpq, bincoef as _bincoef


def bincoef(n, k):
    if k < 0:
        return 0
    else:
        return _bincoef(n, k)


class memo:
    def __init__(self, f):
        self.func = f
        self.cache = {}
        self.highest = 0

    def __call__(self, m):
        if m < self.highest:
            return self.cache[m]
        else:
            x = self.func(m)
            self.cache[m] = x
            self.highest = m
            return x


class cached:
    def __init__(self, f):
        self.func = f
        self.cache = {}
        self.highest = 0

    def __call__(self, m):
        if m in self.cache.keys():
            return self.cache[m]
        else:
            x = self.func(m)
            self.cache[m] = x
            return x


def product(low, high):
    p = 1
    for k in range(low, high+1):
        p *= k
    return p


def A(m, mtop):
    s = 0
    a = bincoef(m+3, m-6)
    for j in range(1, mtop+1):
        s += a*bernoulli(m-6*j)
        a *= product(m-6 - 6*j + 1, m-6*j)
        a //= product(6*j+4, 6*j+9)
    return s


@memo
def b0mod6(m):
    return (mpq(m+3, 3) - A(m, m//6)) / bincoef(m+3, m)


@memo
def b2mod6(m):
    return (mpq(m+3, 3) - A(m, (m-2)//6)) / bincoef(m+3, m)


@memo
def b4mod6(m):
    return (-mpq(m+3, 6) - A(m, (m-4)//6)) / bincoef(m+3, m)


def bernoulli(m):
    assert m >= 0
    if m == 0:
        return 1
    if m == 1:
        return -mpq(1, 2)
    if m % 6 == 0:
        return b0mod6(m)
    if m % 6 == 2:
        return b2mod6(m)
    if m % 6 == 4:
        return b4mod6(m)
    return 0
