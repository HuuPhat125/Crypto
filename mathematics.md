# **1. Quadratic Residues** 
## **Solution**
You can use brute force to solve this.

```python
p = 29
ints = [14, 6, 11]

ans = [x for x in range(p) if(pow(x, 2, p) in ints)]
print(min(ans))
```
# **2. Lengendre Symbol** 
## **Solution**

First, we have to find the quadratic residue (a/p) = a^((p + 1)/2) mod p If (a/p) = 1 then a is quadratic residue Because p = 3 mod 4 so to calculate square root of a, we can use Tonelli–Shanks_algorithm a = x^((p + 1) / 4) mod p

```python
p = 

ints = 

quad = [x for x in ints if(pow(x, (p - 1)//2, p) == 1)]
# there is one quadratic residue
res = quad[0]
print(pow(res, (p + 1)//4, p))
```

# **3. Modular Square Root** 
## **Solution**
```python
def legendre(a, p):
    return pow(a, (p - 1) // 2, p)

def tonelli(n, p):
    assert legendre(n, p) == 1, "not a square (mod p)"
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    if s == 1:
        return pow(n, (p + 1) // 4, p)
    for z in range(2, p):
        if p - 1 == legendre(z, p):
            break
    c = pow(z, q, p)
    r = pow(n, (q + 1) // 2, p)
    t = pow(n, q, p)
    m = s
    t2 = 0
    while (t - 1) % p != 0:
        t2 = (t * t) % p
        for i in range(1, m):
            if (t2 - 1) % p == 0:
                break
            t2 = (t2 * t2) % p
        b = pow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i
    return r

a = 
p = 

root = tonelli(a, p)
print(min(root, p - root))
```

# **4. Chinese Remainder Theorem** 
## **Solution**
```python
from Crypto.Util.number import inverse

def crt(a, m):
    Mul = 1
    for i in m:
        Mul *= i
    M = [Mul // x for x in m]
    y = [inverse(M[i], m[i]) for i in range(len(m))]
    ans = 0
    for i in range(len(m)):
        ans += a[i] * M[i] * y[i]
    return ans % Mul

a = [2, 3, 5]
m = [5, 11, 17]
print(crt(a, m))
```

# **5. Vectors** 
## **Solution**

You can use numpy.array() to solve this

```python
import numpy as np
v = np.array([2, 6, 3])
w = np.array([1, 0, 0])
u = np.array([7, 7, 2])

x = 3*(2*v - w)
y = 2*u

print(x.dot(y))
```

# **6. Size and Basis** 
## **Solution**
You can use numpy.array() to solve this
```python
import numpy as np
v = np.array([4, 6, 2, 5])

print(pow(v.dot(v), 0.5))
```


# **7. Gram Schmidt** 
## **Solution**

use numpy.array() and Gram-Schimidt algorithm presented in the challenge

```python
import numpy as np
v = [np.array([4,1,3,-1]),
     np.array([2,1,-3,4]),
     np.array([1,0,-2,7]),
     np.array([6,2,9,-5])]

u = [v[0]]
for i in range(1, 4):
    mi = [np.dot(v[i], u[j]) / np.dot(u[j], u[j]) for j in range(len(u))]
    u += [v[i] - sum([mij * uj for (mij, uj) in zip(mi, u)])]

print(round(u[3][1], 5))
```
# **8. What's a Lattice** 
## **Solution**
Use numpy.linalg.det() to solve

```python
import numpy as np
v = np.array([[6, 2, -3], [5, 1, 4], [2, 7, 1]])

print(round(abs(np.linalg.det(v))))
```
# **9. Gaussian Reduction** 
## **Solution**
The method to solve this problem is to find the greatest common divisor of two integers using the Euclidean algorithm with NumPy vectors. It continuously subtracts multiples of the smaller number from the larger number until the two numbers are equal and the equal value is the greatest common divisor.
```python
import numpy as np
v1 = np.array([846835985, 9834798552])
v2 = np.array([87502093, 123094980])
m = -1
while(m != 0):
    if (np.dot(v2, v2) < np.dot(v1, v1)):
        t = v1
        v1 = v2
        v2 = t
    m = int((v1.dot(v2)) / (v1.dot(v1)))
    if(m == 0):
        print(v1.dot(v2))
    v2 = v2 - m*v1
```

# **10. Find the Lattice** 
## **Solution**
To solve this problem, we need to create a matrix using the values from public_key and a constant, then apply the LLL algorithm to decipher This code uses decrypt and gauss functions to decrypt a flag encrypted. The decode function takes five integer arguments and performs modular arithmetic to get the decoded message. The gauss function takes two vectors and returns their greatest common divisor. The decoded message is converted to a byte string and printed to the console.
```python
from Crypto.Util.number import long_to_bytes

def decrypt(q, h, f, g, e):
    a = (f*e) % q
    m = (a*inverse_mod(f, g)) % g
    return m

def gauss(v1, v2):
    while True:
        if v2.norm() < v1.norm():
            v1, v2 = v2, v1
        m = round(v1*v2/(v1*v1))
        if m == 0:
            return v1, v2
        v2 = v2-m*v1

h, q = (2163268902194560093843693572170199707501787797497998463462129592239973581462651622978282637513865274199374452805292639586264791317439029535926401109074800, 7638232120454925879231554234011842347641017888219021175304217358715878636183252433454896490677496516149889316745664606749499241420160898019203925115292257)
enc_flag = 5605696495253720664142881956908624307570671858477482119657436163663663844731169035682344974286379049123733356009125671924280312532755241162267269123486523

g = gauss(vector([q,1]),vector([h,1]))[0][0]
f = ZZ(g/GF(q)(h))
print(long_to_bytes(decrypt(q,h,f,g,enc_flag)))
```

# **11. Backpack Cryptography** 
## **Solution**
The code creates a matrix using values from public_key and a constant, and then applies the LLL algorithm to reduce the size of the matrix.
```python
public_key = []
ct = 0
a1, a2, a3, a4 = public_key[:4]
l = 1 / (1 << 380)
m = matrix(QQ, [[l,  a2,  a3,  a4],
                [0, -a1,   0,   0],
                [0,   0, -a1,   0],
                [0,   0,   0, -a1]])
LLLmat = m.LLL()
```

# **12. Successive Powers** 
## **Solution**
This problem finds the smallest prime number satisfying the Chinese Remainder Theorem for a given list s by checking each prime starting at max(s) + 1 and computing a set of values must be equal for the theorem to be true.
```python
from Crypto.Util.number import inverse

s = [588, 665, 216, 113, 642, 4, 836, 114, 851, 492, 819, 237]

pmn = max(s) + 1

for p in range(pmn, 1000):
    x = [(s[i] * inverse(s[i - 1], p)) % p for i in range(1, 12)]
    if(len(set(x)) == 1):
        print(x, p)
        break
```
# **13. Adrien's Signs** 
## **Solution**
The public key for this encryption was provided as two large integers: 
 is a prime number and 
 is a quadratic non-residue modulo 
.

For each 
, compute the Legendre symbol of 
 with respect to 
. If the Legendre symbol is equal to 1, then the corresponding plaintext bit is set to 1; otherwise, it is set to 0.

Convert binary string to ASCII to get the flag.
```python
a = 288260533169915
p = 1007621497415251

ciphertext = [67594220461269, 501237540280788, 718316769824518, 296304224247167, 48290626940198, 30829701196032, 521453693392074, 840985324383794, 770420008897119, 745131486581197, 729163531979577, 334563813238599, 289746215495432, 538664937794468, 894085795317163, 983410189487558, 863330928724430, 996272871140947, 352175210511707, 306237700811584, 631393408838583, 589243747914057, 538776819034934, 365364592128161, 454970171810424, 986711310037393, 657756453404881, 388329936724352, 90991447679370, 714742162831112, 62293519842555, 653941126489711, 448552658212336, 970169071154259, 339472870407614, 406225588145372, 205721593331090, 926225022409823, 904451547059845, 789074084078342, 886420071481685, 796827329208633, 433047156347276, 21271315846750, 719248860593631, 534059295222748, 879864647580512, 918055794962142, 635545050939893, 319549343320339, 93008646178282, 926080110625306, 385476640825005, 483740420173050, 866208659796189, 883359067574584, 913405110264883, 898864873510337, 208598541987988, 23412800024088, 911541450703474, 57446699305445, 513296484586451, 180356843554043, 756391301483653, 823695939808936, 452898981558365, 383286682802447, 381394258915860, 385482809649632, 357950424436020, 212891024562585, 906036654538589, 706766032862393, 500658491083279, 134746243085697, 240386541491998, 850341345692155, 826490944132718, 329513332018620, 41046816597282, 396581286424992, 488863267297267, 92023040998362, 529684488438507, 925328511390026, 524897846090435, 413156582909097, 840524616502482, 325719016994120, 402494835113608, 145033960690364, 43932113323388, 683561775499473, 434510534220939, 92584300328516, 763767269974656, 289837041593468, 11468527450938, 628247946152943, 8844724571683, 813851806959975, 72001988637120, 875394575395153, 70667866716476, 75304931994100, 226809172374264, 767059176444181, 45462007920789, 472607315695803, 325973946551448, 64200767729194, 534886246409921, 950408390792175, 492288777130394, 226746605380806, 944479111810431, 776057001143579, 658971626589122, 231918349590349, 699710172246548, 122457405264610, 643115611310737, 999072890586878, 203230862786955, 348112034218733, 240143417330886, 927148962961842, 661569511006072, 190334725550806, 763365444730995, 516228913786395, 846501182194443, 741210200995504, 511935604454925, 687689993302203, 631038090127480, 961606522916414, 138550017953034, 932105540686829, 215285284639233, 772628158955819, 496858298527292, 730971468815108, 896733219370353, 967083685727881, 607660822695530, 650953466617730, 133773994258132, 623283311953090, 436380836970128, 237114930094468, 115451711811481, 674593269112948, 140400921371770, 659335660634071, 536749311958781, 854645598266824, 303305169095255, 91430489108219, 573739385205188, 400604977158702, 728593782212529, 807432219147040, 893541884126828, 183964371201281, 422680633277230, 218817645778789, 313025293025224, 657253930848472, 747562211812373, 83456701182914, 470417289614736, 641146659305859, 468130225316006, 46960547227850, 875638267674897, 662661765336441, 186533085001285, 743250648436106, 451414956181714, 527954145201673, 922589993405001, 242119479617901, 865476357142231, 988987578447349, 430198555146088, 477890180119931, 844464003254807, 503374203275928, 775374254241792, 346653210679737, 789242808338116, 48503976498612, 604300186163323, 475930096252359, 860836853339514, 994513691290102, 591343659366796, 944852018048514, 82396968629164, 152776642436549, 916070996204621, 305574094667054, 981194179562189, 126174175810273, 55636640522694, 44670495393401, 74724541586529, 988608465654705, 870533906709633, 374564052429787, 486493568142979, 469485372072295, 221153171135022, 289713227465073, 952450431038075, 107298466441025, 938262809228861, 253919870663003, 835790485199226, 655456538877798, 595464842927075, 191621819564547]

plaintext = ''.join([ '1' if pow(c, (p - 1) // 2, p) == 1 else '0' for c in ciphertext])

print( ''.join( [ chr(int(plaintext[i:i+8], 2)) for i in range(0, len(plaintext), 8)] ) )
```
# **14. Modular Binomials** 
## **Solution**
We need to factorize an modulus given two encrypted messages and their corresponding public keys 
 and 
. Then, compute two values 
 and 
, which are related to the two ciphertexts and can be used to derive the common factor 
 of the modulus 
. After computing 
, it obtains the second factor 
 by dividing 
 by 
.
```python
from math import gcd

n = 14905562257842714057932724129575002825405393502650869767115942606408600343380327866258982402447992564988466588305174271674657844352454543958847568190372446723549627752274442789184236490768272313187410077124234699854724907039770193680822495470532218905083459730998003622926152590597710213127952141056029516116785229504645179830037937222022291571738973603920664929150436463632305664687903244972880062028301085749434688159905768052041207513149370212313943117665914802379158613359049957688563885391972151218676545972118494969247440489763431359679770422939441710783575668679693678435669541781490217731619224470152467768073
e1 = 12886657667389660800780796462970504910193928992888518978200029826975978624718627799215564700096007849924866627154987365059524315097631111242449314835868137
e2 = 12110586673991788415780355139635579057920926864887110308343229256046868242179445444897790171351302575188607117081580121488253540215781625598048021161675697
c1 = 14010729418703228234352465883041270611113735889838753433295478495763409056136734155612156934673988344882629541204985909650433819205298939877837314145082403528055884752079219150739849992921393509593620449489882380176216648401057401569934043087087362272303101549800941212057354903559653373299153430753882035233354304783275982332995766778499425529570008008029401325668301144188970480975565215953953985078281395545902102245755862663621187438677596628109967066418993851632543137353041712721919291521767262678140115188735994447949166616101182806820741928292882642234238450207472914232596747755261325098225968268926580993051
c2 = 14386997138637978860748278986945098648507142864584111124202580365103793165811666987664851210230009375267398957979494066880296418013345006977654742303441030008490816239306394492168516278328851513359596253775965916326353050138738183351643338294802012193721879700283088378587949921991198231956871429805847767716137817313612304833733918657887480468724409753522369325138502059408241232155633806496752350562284794715321835226991147547651155287812485862794935695241612676255374480132722940682140395725089329445356434489384831036205387293760789976615210310436732813848937666608611803196199865435145094486231635966885932646519

q1 = pow(c1, e2, n)
q2 = pow(c2, e1, n)
d = pow(5, e1 * e2, n) * q1 - pow(2, e1 * e2, n) * q2
q = gcd(d, n)
p = n // q
print("p=",p)
print("q=",q)
```

# **15. Broken RSA** 
## **Solution**
Because the ciphertext has a small encryption exponent, we can decrypt the ciphertext by repeatedly taking square roots modulo 
```python
from sympy.ntheory.residue_ntheory import sqrt_mod

n = 27772857409875257529415990911214211975844307184430241451899407838750503024323367895540981606586709985980003435082116995888017731426634845808624796292507989171497629109450825818587383112280639037484593490692935998202437639626747133650990603333094513531505209954273004473567193235535061942991750932725808679249964667090723480397916715320876867803719301313440005075056481203859010490836599717523664197112053206745235908610484907715210436413015546671034478367679465233737115549451849810421017181842615880836253875862101545582922437858358265964489786463923280312860843031914516061327752183283528015684588796400861331354873
ct = 11303174761894431146735697569489134747234975144162172162401674567273034831391936916397234068346115459134602443963604063679379285919302225719050193590179240191429612072131629779948379821039610415099784351073443218911356328815458050694493726951231241096695626477586428880220528001269746547018741237131741255022371957489462380305100634600499204435763201371188769446054925748151987175656677342779043435047048130599123081581036362712208692748034620245590448762406543804069935873123161582756799517226666835316588896306926659321054276507714414876684738121421124177324568084533020088172040422767194971217814466953837590498718

for a in sqrt_mod(ct, n, all_roots=True):
    for b in sqrt_mod(a, n, all_roots=True):
        for c in sqrt_mod(b, n, all_roots=True):
            for d in sqrt_mod(c, n, all_roots=True):
                try:
                    print(bytes.fromhex(hex(d)[2:]).decode())
                except:
                    continue
```
# **16. No Way Back Home** 
## **Solution**
```python
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from hashlib import sha256
from Crypto.Util.number import long_to_bytes

p, q = (10699940648196411028170713430726559470427113689721202803392638457920771439452897032229838317321639599506283870585924807089941510579727013041135771337631951, 11956676387836512151480744979869173960415735990945471431153245263360714040288733895951317727355037104240049869019766679351362643879028085294045007143623763)
vka = 124641741967121300068241280971408306625050636261192655845274494695382484894973990899018981438824398885984003880665335336872849819983045790478166909381968949910717906136475842568208640203811766079825364974168541198988879036997489130022151352858776555178444457677074095521488219905950926757695656018450299948207
vkakb = 114778245184091677576134046724609868204771151111446457870524843414356897479473739627212552495413311985409829523700919603502616667323311977056345059189257932050632105761365449853358722065048852091755612586569454771946427631498462394616623706064561443106503673008210435922340001958432623802886222040403262923652
vkb = 6568897840127713147382345832798645667110237168011335640630440006583923102503659273104899584827637961921428677335180620421654712000512310008036693022785945317428066257236409339677041133038317088022368203160674699948914222030034711433252914821805540365972835274052062305301998463475108156010447054013166491083
c = 'fef29e5ff72f28160027959474fc462e2a9e0b2d84b1508f7bd0e270bc98fac942e1402aa12db6e6a36fb380e7b53323'

n = p * q
rka = vka // p
rkakb = vkakb // p
k_B = (rkakb * pow(rka, -1, q)) % q
v = (vkb * pow(k_B, -1, n)) % n

key = sha256(long_to_bytes(v)).digest()
cipher = AES.new(key, AES.MODE_ECB)
pt = cipher.decrypt(bytes.fromhex(c))
print(unpad(pt, 16)
```

# **17. Ellipse Curve Cryptography** 
## **Solution**
To solve this problem, perform an elliptic curve encryption challenge that involves transferring data between two parties, solving the discrete logarithm problem, obtaining a shared secret key, and using that key to decrypt encryption flags.
```python
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from hashlib import sha1
from collections import namedtuple
Point = namedtuple("Point", "x y")

def point_addition(P, Q):
    Rx = (P.x*Q.x + D*P.y*Q.y) % p
    Ry = (P.x*Q.y + P.y*Q.x) % p
    return Point(Rx, Ry)

def scalar_multiplication(P, n):
    Q = Point(1, 0)
    while n > 0:
        if n % 2 == 1:
            Q = point_addition(Q, P)
        P = point_addition(P, P)
        n = n//2
    return Q

def gen_shared_secret(P, d):
    return scalar_multiplication(P, d).x

p = 173754216895752892448109692432341061254596347285717132408796456167143559
D = 529
Dsqrt = 23 # 23^2 = 529
G = Point(29394812077144852405795385333766317269085018265469771684226884125940148,
          94108086667844986046802106544375316173742538919949485639896613738390948)
A = Point(155781055760279718382374741001148850818103179141959728567110540865590463,
          73794785561346677848810778233901832813072697504335306937799336126503714)
B = Point(171226959585314864221294077932510094779925634276949970785138593200069419,
          54353971839516652938533335476115503436865545966356461292708042305317630)

g = G.x - Dsqrt*G.y
h = A.x - Dsqrt*A.y
n_a = discrete_log(GF(p)(h), GF(p)(g))

shared_secret = gen_shared_secret(B, n_a)
key = sha1(str(shared_secret).encode('ascii')).digest()[:16]
iv = bytes.fromhex('64bc75c8b38017e1397c46f85d4e332b')
encrypted_flag = bytes.fromhex('13e4d200708b786d8f7c3bd2dc5de0201f0d7879192e6603d7c5d6b963e1df2943e3ff75f7fda9c30a92171bbbc5acbf')
cipher = AES.new(key, AES.MODE_CBC, iv)
flag = unpad(cipher.decrypt(encrypted_flag), 16).decode()
print(f'FLAG: {flag}'
```

# **18. Roll your Own** 
## **Solution**
By asking @Giapp, to solve this problem, we need establishes a telnet connection to the server, reads and sends data in JSON format, does some calculations, and prints the flags received from the server.
```python
import telnetlib
import json

r = telnetlib.Telnet("socket.cryptohack.org", 13403)

def readline():
    return r.read_until(b"\n")

def json_send(hsh):
    request = json.dumps(hsh).encode()
    r.write(request)

q = readline().split()[-1].decode()[1:-1]
q = int(q, 16)

g = q+1
n = q**2
json_send({"g":hex(g), "n":hex(n)})

h = readline().split()[-1].decode()[1:-1]
h = int(h, 16)

x = (h-1)//q

json_send({"x":hex(x)})
print(readline().decode())
```
# **19. Unencryptable** 
## **Solution**
```python
from Crypto.Util.number import *

N = '0x7fe8cafec59886e9318830f33747cafd200588406e7c42741859e15994ab62410438991ab5d9fc94f386219e3c27d6ffc73754f791e7b2c565611f8fe5054dd132b8c4f3eadcf1180cd8f2a3cc756b06996f2d5b67c390adcba9d444697b13d12b2badfc3c7d5459df16a047ca25f4d18570cd6fa727aed46394576cfdb56b41'
e = '0x10001'
c = '0x5233da71cc1dc1c5f21039f51eb51c80657e1af217d563aa25a8104a4e84a42379040ecdfdd5afa191156ccb40b6f188f4ad96c58922428c4c0bc17fd5384456853e139afde40c3f95988879629297f48d0efa6b335716a4c24bfee36f714d34a4e810a9689e93a0af8502528844ae578100b0188a2790518c695c095c9d677b'

N, e, c = int(N,16), int(e,16), int(c,16)

DATA = bytes.fromhex("372f0e88f6f7189da7c06ed49e87e0664b988ecbee583586dfd1c6af99bf20345ae7442012c6807b3493d8936f5b48e553f614754deb3da6230fa1e16a8d5953a94c886699fc2bf409556264d5dced76a1780a90fd22f3701fdbcb183ddab4046affdc4dc6379090f79f4cd50673b24d0b08458cdbe509d60a4ad88a7b4e2921")
data = bytes_to_long(DATA)

encrypted_data = pow(data,e,N)
assert encrypted_data == data

for i in range(1,17):
	data = (data * data) % N
	print(data , i)

z = 14573330329777345466716427138964763643316382540146850673261646339635250550047603089851590671733675671553339810312988446921139294433175492148910522081441119373450446646232663100181438157456101244865003689391419997679731752504790195214890611503266822839198431068665139221007788734481289099967859381061489428771
assert (z * z) % N == 1

p = GCD(z-1 , N)
q = GCD(z+1 , N)

assert p * q == N

d = inverse(e, phi)
flag = pow(c, d , N)
flag = long_to_bytes(flag)
print(flag)
```

# **20. Cofactor Cofantasy** 
## **Solution**
A tricky way is that when the i-th bit is 1 server try exponential operation with random exponent. Otherwise, server will just send random value. It's pretty obvious that picking random and doing exponential operation is slower than just picking random. So, We can do side channel attack with that fact.
```python
import time
from telnetlib import Telnet
import json
from statistics import median,mean
from tqdm import tqdm,trange

def two_clursturing(datas, epoch=10):
    centor=[min(datas),max(datas)]
    label=[0]*len(datas)
    for _ in range(epoch):
        bag=[[],[]]
        for i in range(len(datas)):
            if abs(datas[i]-centor[0])<abs(datas[i]-centor[1]):
                label[i]=0
                bag[0].append(datas[i])
            else:
                label[i]=1
                bag[1].append(datas[i])
        centor[0]=mean(bag[0])
        centor[1]=mean(bag[1])
        centor.sort()
    return label

cli=Telnet("socket.cryptohack.org",13398)

print(cli.read_until(b"\n"))

# Higher = slower&better
precision=10

found=b""
pbar = trange(0*8,43*8,8)
for i in pbar:
    val=0
    ssamp=[]
    for j in trange(8,leave=False):
        sample=[]
        query={"option":"get_bit","i":i+j}
        eq=json.dumps(query).encode()
        for _ in range(precision):
            st=time.time_ns()
            cli.write(eq)
            cli.read_until(b"\n")
            ed=time.time_ns()
            sample.append(ed-st)
        ssamp.append(median(sample))

    b="".join(map(str,two_clursturing(list(reversed(ssamp)))))
    found+=bytes([int(b,2)])
    pbar.set_description(str(found))

print(found)
```

# **21. Real Eisenstein** 
## **Solution**
My solution is similar to others, but uses python + PARI/GP, instead of Sage. Note that Pari/GP's LLL returns the transformation matrix, instead of LLL-reduced basis.
```python
import cypari2
import math
from decimal import *
getcontext().prec = 100

pari = cypari2.Pari()

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
cipher = 1350995397927355657956786955603012410260017344805998076702828160316695004588429433

PRIMES_sqrt_pow2 = list(map(lambda x:Decimal(x).sqrt() * (16**64), PRIMES))

n = len(PRIMES)
C = int(Decimal(n).sqrt()) + 1

mat = pari.matrix(n+1, n+1)
for i in range(n):
    mat[i, i] = 1
    mat[n, i] = -C * PRIMES_sqrt_pow2[i]
mat[n, n] = C * cipher

trans_L = pari.qflll(mat)
print(''.join(list(map(chr, trans_L[0]))))
```