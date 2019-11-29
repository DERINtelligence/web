---
layout: post
title: Təbii dilin emalı - Giriş, TF-IDF, n-gram.
author: Mammad Hajili
---

Təbii dilin kompüter tərəfindən emalı ilkin hesablayıcı maşınlar erasından bəri ən önəmli ideyalardan biri idi. Belə ki, hətta Stenli Kübrikin "2001: A Space Odyssey" filmində HAL 9000 kompüteri ingilis dilini anlayır və danışır kimi təsvir edilmişdi. Təbii ki, filmin nümayiş tarixi 1968-cı ili nəzərə aldığımızda izləyicilər bütün elmi-fantastika filmlərində olduğu kimi bu filmdə də reallıqdan uzaq optimistik bir hal ilə qarşı-qarşıya qalmışdılar. Ancaq süni intellekt anlayışının kompüter elmləri və fəlsəfi nəşrlərdə üzə çıxması və bu sahələrdə olan elmi araşdırmalar (maşın öyrənmə, proqramlaşdırma dilləri və neyroelmlər) iki sualın verilməsinə gətirib çıxarırdı.

1. Yazı və ya danışıqdan kompüterin riyazi əməliyyatlar yerinə yetirə biləcəyi və üzərində araşdıracağı bir riyazi anlayış qurmaq mümkündür mü?
2. Kompüterdəki hər hansı riyazi təsvirdən yazı və ya danışıq yaradıla bilər mi?

Bu iki sual süni intellekt üçün riyazi dilçilik və müasir maşın öyrənmə konteksində hələ də cavablanmağa çalışan suallardır. Bu iki sualı cavablamağa çalışan elm sahəsi isə **təbii dilin emalı** (ing. Natural Language Processing - NLP) adlanır.

Belə bir problem var ki, biz insanlığın bütün təbii dillərini və insanların düşüncələrini (hələ ki) tamamilə modelləyə bilmirik. Bunun əvəzinə isə müasir NLP-in işləmə prinsipi əsas etibarı ilə spesifik tapşırıqlar üçün verilən datadan öyrənmə, ona uyğun riyazi model qurma və modelin yekun qiymətləndirilməsinə əsaslanır.

Bu yazıda(ümid edirəm, yazılarda) mən təbii dillərin yalnızca yazılı təsvirinin emalı haqqında danışacam. Belə ki, bundan sonra "təbii dilin emalı" söz birləşməsinin əvəzinə NLP abbreviaturasından istifadə olunacaq və haqqında danışılacaq mövzular və alqoritmlər yazının emalı haqqında olacaqdır.

NLP-nin istifadə olunduğu və üzərində hələ də elmi araşdırmalar gedən bəzi tapşırıqlar bunlardır:

- Maşın tərcüməsi (Machine Translation)
- Sual cavablama və ya mətn özətləmə (Questions Answering, summarization)
- Məlumat axtarışı və əldə edilməsi (Information search, retrieval, extraction)
- Məlumatın filtri və sinifləndirilməsi (Information filtering and classification)

Yuxarıda da qeyd etdiyim kimi bunların hər biri spesifik tapşırıqlardır və hər biri üçün uyğun datasetlər və riyazi modellər vacibdir. Ancaq bu tapşırıqların həlləri maşın öyrənmə alqoritmi mövcud olsa belə çox da asan deyil. Çünki, lazım olan datasetin əldə edilməsi heç də ucuz və rahat başa gələn bir iş deyil. Əksər tapşırıqlarda işarələnmiş data lazımlıdır, ancaq bu data bəzi hallarda internetdə məlumat bazalarında ya mövcud olmur, ya da arzuolunan şəkildə olmur. Mövcud olmayan datasetin yaradılması üçün lazım olan tekstin işarələnməsi isə diqqətlə və düzgün edilməli iş olduğu üçün tapşırığın aid olduğu sahənin ekspertlərin köməyi lazımdır. Bu isə əlavə pul və vaxt xərci ilə nəticələnir. Məsələn, tapşırıq verilən cümlənin cümlə üzvlərinin işarələnməsidirsə, bizə lazım olan data çoxlu cümlələr və cümlədəki hər bir sözə/birləşməyə uyğun cümlə üzvünün işarələnməsidir. Təbii ki, dilçilik sahəsində ekspert olmayan biri nə qədər çalışsa da, çox güman ki, hər bir istisna halı nəzərə almayacaq və çoxlu səhvlər buraxacaq, buna görə də belə datasetlər hazırlandığında sahə üzrə təcrübəsi olan insanların köməyindən istifadə edilir. Daha bir çətinlik isə, yenə əksər hallarda hər hansı bir dil üçün, məsələn, ingilis dilinin cümlə üzvləri üçün hazırlanan model başqa bir dil, məsələn, Azərbaycan dili üçün yararlı olmamasıdır. Buna görə də tapşırıqların həllinin çətinliyi və ya mümkünlüyü datasetin mövcudluğundan və ölçüsündən asılı olaraq müxtəlif dillər üçün də fərqlənir.

Datasetin olması yaxşı xəbər olsa da, kompüterin onu anlaması üçün teksti riyazi bir anlayış ilə ifadə etməyimiz vacibdir. NLP-də bu proses **tekstin vektorizasiyası** adlanır. Yəni biz yazıdakı hər bir sözü, dolayısı ilə cümləni, paraqrafı və mətni vektor ilə ifadə edirik. Tək-tək işarələr məna ifadə etmədiyindən bəzi işarə əsaslı neyron modelləri çıxmaq şərti ilə əsasən NLP-də ən kiçik vahid kimi ard-arda gələn işarələr toplusunu qəbul edirik. NLP-də belə toplu **token** adlanır. Tokenlər əsasən sözlər olsa da, bəzən istisna hallarla qarşılaşdığımız üçün belə topluları sadəcə "söz" adlandırmırıq. Datasetdəki bütün fərqli tokenlərdən ibarət olan set **lüğət** adlanır.

**Unitar Kod**

Tekstin ən sadə vektorizasiyası [**unitar kod**](https://en.wikipedia.org/wiki/One-hot) üsuludur. Belə ki, belə vektorların ölçüsü hər bir token üçün lüğətin ölçüsünə bərabərdir və tokenin lüğətdəki yerini ifadə edən indeks 1 ilə, qalan bütün indekslər 0 ilə ifadə olunur. Gəlin hesab edək bizim kiçik datasetimiz var və bu dataset aşağıdakı 2 tekstdən ibarətdir.

1. Dostum kitab oxumağı sevir. O, filmləri də sevir.
2. Kitab masada qalıb, dostu gətirəcək.

Lüğət: 
```
'Dostum', 'dostu', 'də', 'filmləri', 'gətirəcək', 'Kitab',
'kitab', 'masanın', 'qalıb', 'O', 'oxumağı', 'sevir', ',', '.'
```

Tokenlərin lüğətdəki yerini nəzərə alarsaq, bəzi tokenlərin unitar kod vektorları belə olacaqdır:
- ```"dostu" : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]```
- ```"sevir" : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]```

Bu datasetdə lüğətin ölçüsü 14-dür. Belə halda yuxarıda da gördüyümüz kimi hər bir unitar kod vektorun ölçüsü (14,1) olacaqdır. Bütün lüğəti ifadə etmək istəsək bizə (14,14) ölçülü vahid matris lazım olacaqdır. Nəzərə alsaq ki, böyük datasetlər üçün lüğət də olduqca böyük olacaq və bu halda vahid matris olduqca seyrək olacaqdır. Belə seyrək böyük ölçülü matrislər bizə öyrənmə alqoritmlərindəki matrislərin vurulması, tərsinin tapılması əməliyyatları zamanı çoxlu vaxt və resurs itkisinə səbəb olacaqdır. Təbii ki, bu problemləri aradan qaldırmağın yolları vardır. Bunlardan ilki lüğətin mümkün olduqca ölçüsünü azaltmaqdır. Bunun üçün müxtəlif üsullar vardır, bu üsullar toplusuna **tekstin təmizlənməsi və ilkin prosesi** (text cleaning and preprocessing) deyilir.

Bizim kiçik misalımızda olan lüğətə fikir versək, görərik ki, kökü eyni olan sözlərin hər biri fərqli bir tokenlə ifadə olunub. ("insana, insanların"). Həmçininin durğu işarəsi və ədat kimi kontekst ifadə etməyən tokenlər də lüğətdə mövcuddur. Hətta biri böyük və digəri kiçik hərflə başlayan eyni söz fərqli vektorlarla ifadə olunur. Bunları və digər bizim misallarda olmayan halları ehtiva edən üsulları gəlin incələyək:

**Təmizlik**

Təmizlik əsasən internetdən əldə etdiyimiz yazılı məlumatların normal cümlə strukturu halına salmaq üçün istifadə olunur. Bu metod üçün əsasən **regex ifadələrindən** istifadə olunur. Mən bəzi halları Python istifadə edərək göstərməyə çalışacam.
    
1. Bəzən tekstlər müxtəlif internet bazalarından əldə edildikdə onların içində xüsusi HTML və ya XML "tag"-lərdən istifadə olunur. Məsələn:
    
    ```html
    <b> çox gözəl məshuldur </b>
    ```
    Buradakı ```<b>``` və oxşar tag-ləri silmək üçün 
    istifadə olunan regex ifadəsi budur:
    ```python
    import re

    text = "<b> çox gözəl məshuldur </b>"
    clean_text = re.sub(re.compile('<.*?>'), '', text)
    print(clean_text)
    
    """
        Çıxış: 
            çox gözəl məshuldur
    """
   ```
    Bu metod tvitlərdən ibarət olan datasetdə linklərin, istifadəçi adlarının silinməsi üçün də istifadə oluna bilər. Regex ifadələr ilə tanış olub bu hallar üçün özünüz yoxlaya bilərsiniz.

2. Durğu işarələrinin silinməsini də regex ifadələrdən istifadə edərək edə bilərik.
    ```python
    import re
    
    text = "Biz İsmayıllı, Şəki və Qəbələdə olduq."
    
    clean_text = re.sub('[\W_]+', ' ', text, re.UNICODE)
    
    print(clean_text)
    
    """
        Çıxış: 
            Biz İsmayıllı Şəki və Qəbələdə olduq
    """
    ```


**Tekstin ilkin prosesi**

1. Kiçik hərflər

    Bizim yuxarıda işlətdiyimiz iki cümləlik kiçik datasetdə "kəndlərdə" və "Kəndlərdə" fərqli tokenlər idi. Bu hal böyük datasetlərdə lüğətin ən az iki dəfə artmasına gətirib çıxarır(bəzi hallarda səhv yazılışlar nəticəsində bu əmsal daha da arta bilər), həm də məna etibarı ilə bu iki söz eynidir. Ona görə də tekstin ilkin prosesi zamanı bütün hərflərin kiçiyə çevirmək çox önəmlidir. Bu əməliyyat üçün Python ```.lower()``` funksiyasından istifadə edə bilərik.
    
    ```python
    import re

    text = "Biz İsmayıllı, Şəki və Qəbələdə olduq."

    #durğu işarələrinin silinməsi
	clean_text = re.sub('[\W_]+', ' ', text, re.UNICODE)
    
    clean_text = clean_text.replace('İ', 'I')
    clean_text = clean_text.lower()

    print(clean_text)
    
    """
        Çıxış:
            biz ismayıllı şəki və qəbələdə olduq
    """
    ```
2. Lazımsız sözlərin(Stopwords) silinməsi

    Lazımsız sözlərin silinməsi olduqca önəmli üsuldur, belə ki, bəzi sözlər həddindən artıq çox və ya nadir istifadə edildiyindən ümumi tekst üçün mənaya çox təsiri olmur, üstəlik lüğətin ölçüsünü böyüdür. Əksər hallarda lazımsız sözlərin siyahısı [Zipf qanununa](https://en.wikipedia.org/wiki/Zipf%27s_law)  əsasən böyük toplular üçün olan statistikalar ilə hazırlanır. Əsasən bu siyahıya "da/də" kimi ədatlar, "o/sən/mən" kimi əvəzliklər, bağlayıcılar, çox işlənən isimlər və fellər aid olur. Siyahılardan biri ilə bu [linkdən](https://github.com/domspad/azerbaijani_stop_words/blob/master/azerwords.csv) tanış ola bilərsiniz.
    ```python
    import re

    text = "Biz İsmayıllı, Şəki və Qəbələdə olduq."
    
    #burada lazımsız sözlərin siyahısı olmalıdır. məs. [biz, və]
    stopwords = #siyahı
    
    #durğu işarələrinin silinməsi
    clean_text = re.sub('[\W_]+', ' ', text, re.UNICODE)   
    clean_text = clean_text.replace('İ', 'I')

    #kiçik hərfə çevirmə
    clean_text = clean_text.lower() 
    #başdakı və sondakı boşluqları silmə
    clean_text = clean_text.strip()
    
    #cümlənin tokenlərə ayrılması
    tokens = clean_text.split(' ') 
    
    #stopwords listində olmayan tokenlər
    tokens = [token for token in tokens if token not in stopwords]
    
    print(tokens)
    
    """
        Çıxış: 
            ["ismayıllı", "şəki", "qəbələdə", "olduq"]
    """
    ```

3. Sözün kökünü alma(Stemming)

    Azərbaycan dili üçün olan ən önəmli üsul sözün kökünü alma üsuludur. Belə ki, dilimiz aqqlütinativ dildir və dilimizdə çoxlu sözdüzəldici şəkilçilər var. Bu şəkilçilər yeni bir məna yaratmır, sadəcə cümlə quruluşuna əsasən sözə uyğun vəziyyət gətirir. Bizim kiçik datasetdə "insanlar" və "insanların" buna bir misal ola bilər. Kökünə alma üsulundan istifadə edərək hər ikisini "insan" tokeni ilə əvəz edə bilərik. Belə ki, bu üsul lüğətin ölçüsünü də kəskin şəkildə kiçildəcəkdir. NLP-də sözünü kökünü alma üçün ən məşhur və önəmli alqoritmlərdən biri Porter alqoritmidir. Azərbaycan dili üçün bu alqoritmin koduna bu [keçiddən](https://github.com/aznlp/azerbaijani-language-stemmer) baxa bilərsiniz. Kodun istifadəsi üçün aşağıdakı əmrləri yerinə yetirməlisiniz:
   
   - Əgər ```git``` - dən istifadə edirsinizsə, ```git clone https://github.com/aznlp/azerbaijani-language-stemmer.git``` - ı əmrini istifadə edərək, etmirsinizsə, keçiddən ```zip``` formatında endirib, daha sonra qovluğa çıxarış edə bilərsiniz.
   - Endirilən qovluqda yeni bir Python file yaradıb aşağıdakı kod ilə test edə bilərsiniz: 

    ```python
    from stemmer import Stemmer
    import re

    text = "Biz İsmayıllı, Şəki və Qəbələdə olduq."
    
    #burada lazımsız sözlərin siyahısı olmalıdır. məs. [biz, və]
    stopwords = #siyahı
    
    #durğu işarələrinin silinməsi
    clean_text = re.sub('[\W_]+', ' ', text, re.UNICODE)
    clean_text = clean_text.replace('İ', 'I')
    
    #kiçik hərfə çevirmə
    clean_text = clean_text.lower() 
    #başdakı və sondakı boşluqları silmə
    clean_text = clean_text.strip()
    
    #cümlənin tokenlərə ayrılması
    tokens = clean_text.split(' ') 
    
    #stopwords listində olmayan tokenlər
    tokens = [token for token in tokens if token not in stopwords]
    
    stemmer = Stemmer()
    
    # tokenlərin köklərinin alınması
    stems = stemmer.stem_words(tokens) 
    
    print(stems)
    '''
       Çıxış: 
           ["ismayıllı", "şəki", "qəbələ", "ol"] 
    '''
    ```
    
Sadaladığımız üsullardan istifadə edərək əvvəl göstərdiyimiz 2 tekstdən ibarət olan kiçik datasetimiz yenidən incələyək:

1. Dostum kitab oxumağı sevir. O, filmləri də sevir.
2. Kitab masada qalıb, dostu gətirəcək.

Təmizlik və ilkin prosesdən sonra lüğətimiz belə olacaqdır:

```'dost', 'film', 'gətir', 'kitab', 'masa', 'oxu', 'sev', 'qal'```

Tokenlərin lüğətdəki yenidən yerini nəzərə alarsaq, bəzi tokenlərin unitar kod vektorları belə olacaqdır:

- ```dost - [1, 0, 0, 0, 0, 0, 0, 0]```
- ```masa - [0, 0, 0, 0, 1, 0, 0, 0]```

Fikir versək, görərik ki, təmizlik və ilkin prosesdən sonra ```'.', ',' 'də', 'O'``` silinmiş, ```'Dostum'``` və ```'dostu'``` isə ```'dost'``` kökü isə əvəz olunmuşdur. Hər bir unitar kod vektor isə lüğətin ölçüsü kiçildiyi üçün (8, 1) ölçülü olacaqdır. Hər bir teksti vektor şəklində göstərmək üçün tokenlərin vektor cəmini almaq kifayət edir. Yəni bu o deməkdir ki, tekstdəki mövcud tokenlərin indesksi vektorda 1 ilə, lüğətdəki digər tokenlərin indeksləri isə 0 ilə ifadə olunur.

```1. ["dost", "kitab", "oxu", "sev", "film", "sev"] - [1, 1, 0, 1, 0, 1, 1, 0]```
```2. ["kitab", "masa", "qal", "dost", "gətir"] - [1, 0, 1, 1, 1, 0, 0, 1]```

Haqqında danışdığımız unitar kod modelinə fikir versək, görərik ki, tekstdəki sözlərin sırası vektorizasiya zamanı önəm daşımır. Belə modellərə NLP-də [**söz çantası**](https://en.wikipedia.org/wiki/Bag-of-words_model) modelləri deyilir. Hər bir vektorun 1 olan indekslərinə uyğun gələn tokenlər birlikdə bir multiset və ya başqa cür desək çantanı əmələ gətirir. Məsələn yuxarıda da göstərdiyim kimi birinci teksti ifadə edən ```[1, 0, 1, 1, 1, 0, 0, 1]``` vektoruna uyğun gələn multiset/çanta ```["kitab", "masa", "qal", "dost", "gətir"]``` olacaqdır.

**Termin tezliyi**

Unitar kod modeli çox sadə olsa da, tekstdəki tokenlərin tezliyini ifadə edə bilmir. Məsələn, incələdiymiz tekstlərdən birinə baxaq:

```Dostum kitab oxumağı sevir. O, filmləri də sevir.``` 

Burada təmizlik və ilkin prosesdən sonra "sev" tokeni iki dəfə təkrarlanır, ancaq bizim vektorumuz unitardır, yəni tokenin neçə dəfə təkrarlanmağından asılı olmayaraq vektorda 0 və ya 1 dəyərini alır. Bu vəziyyət bəzi hallarda problem ola bilər, belə ki, sözün bir neçə dəfə işlənməsi ümumi məzmuna daha çox təsir etməsi ilə nəticələnə bilər. Unitar kod vektorun bu əskikliyini aradan qaldırmaq üçün oxşar model olan [**termin tezliyindən**](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)(TF) istifadə olunur. Belə ki, bu vektor üçün hər bir tokenin tekstdə olub-olmamasından əlavə, neçə dəfə rast gəlindiyi də vacibdir. Deməli, bayaq vurğuladığımız tekst üçün TF vektoru belə ifadə edə bilərik:

``` txt
["dost", "kitab", "oxu", "sev", "film", "sev"]
{"dost":1, "kitab":1, "oxu":1, "sev":2, "film":1}
[1, 1, 0, 1, 0, 1, 2, 0]
```

Əksər hallarda TF vektoru normalizasiya edildikdən sonra işlədilir. Bunun üçün bir çox üsul olsa, onların içində ən çox işlədiləni tekstdəki tokenlərin ümumi sayına nisbəti ilə normallaşdırmadır. Bizim misala diqqət etsək, tekst 6 (1 + 1 + 1 + 2 + 1 = 6) tokendən ibarətdir. Deməli bizim vektorumuz ```tf = [1/6, 1/6, 0, 1/6, 0, 1/6, 1/3, 0]```  olacaqdır.

Termin tezliyinin unitar koda görə digər müsbət cəhətlərindən biri də tekstdəki hər hansı bir tokeni vektor şəklində yox, real ədədlərlə ifadə edə bilməyimizdir. Yəni, üzərində işlədiyimiz misalda teksti d ilə işarə etsək, ```tf("dost", d) = 1/6``` olduğunu görə bilərik. Halbuki, unitar kod vektorunda bunun üçün lüğətin uzunluğu ölçüsündə vektordan istifadə edirdik. 

Termin tezliyi unitar koda görə olduqca yaxşı olsa da, fikir versək, görə bilərik ki, hər hansı tekstdəki bir token üçün olan dəyər yalnız və yalnız o tekstdəki tokenlərin işlənməsindən və sayından asılıdır. Məsələn, ümumi datasetdə çox nadir olan və ixtiyari bir tekstdə 1 dəfə işlənən bir token, həmin tekstdə 1 dəfə işlənən başqa sözlə eyni TF dəyərinə malikdir. Ancaq, bu nadir token işləndiyi tekstin əsas mənasını ehtiva edə bilər. Bu problemi aradan qaldırmaq üçün istifadə olunan üsul isə **tərs dokument tezliyi** (IDF) adlanır. Bir token üçün IDF dəyəri datasetdəki tekstlərin ümumi sayının həmin token işlənən tekstlərin sayına nisbətinin loqarifmik dəyərinə bərabərdir. 

$$ \mathrm{idf}(token, D) = \log \dfrac{|D|}{|\{d \in D: token \in d\}|}$$

Burada D datasetdəki tekstlərin/dokumentlərin siyahısıdır. Hesablamalar zamanı TF və IDF dəyərlərinin hasilindən istifadə edirik və bu dəyərə **TF-IDF** dəyəri deyilir.

$$ \mathrm{tf\_idf}(token, d, D) = \mathrm{tf}(token, d) * \mathrm{idf}(token, D) $$

Tekstin TF-IDF vektorunu almaq üçün TF vektorunu vektordakı tokenlərin uyğun IDF dəyəri ilə hasilini tapmaq kifayətdir. Buna uyğun olaraq kiçik misalımıza  bir də diqqət yetirək:

TF vektorlar aşağıdakılardır:
``` txt
1. ["dost", "kitab", "oxu", "sev", "film", "sev"]
[1/6, 1/6, 0, 1/6, 0, 1/6, 1/3, 0]
2. ["kitab", "masa", "qal", "dost", "gətir"]
[1/5, 0, 1/5, 1/5, 1/5, 0, 0, 1/5]
```

Hər bir token üçün IDF dəyərlər:

$$
	\begin{aligned}
	    \mathrm{idf}(kitab) &= \mathrm{idf}(dost) = \log {2 \over 2} = 0\\
	    \mathrm{idf}(oxu) \, & = \mathrm{idf}(sev) = 
	    \mathrm{idf}(film) = \\ & \qquad = \mathrm{idf}(qal) = 
	    \mathrm{idf}(gətir) = \\ & \qquad = \mathrm{idf}(masa) = 
	    \log {2 \over 1} \approx 0.7
	\end{aligned}    
$$

Yekun olaraq hər bir tekst üçün TF-IDF vektorlar belə olacaqdır:

```txt
tf_idf("Dostum kitab oxumağı sevir. O, filmləri də sevir.") =
        = [1/6*0, 1/6*0.7, 0, 1/6*0, 0, 1/6*0.7, 1/3*0.7, 0] = 
        = [0, 0.115, 0, 0, 0, 0.115, 0.23, 0]
```
```txt
tf_idf("Kitab masada qalıb, dostu gətirəcək.") =
    = [1/5*0, 0, 1/5*0.7, 1/5*0, 1/5*0.7, 0, 0, 1/5*0.7] =
    = [0, 0, 0.14, 0, 0.14, 0, 0, 0.14]
```

***n*-gram**

Biz incələdiyimiz hər iki modeldə də yalnızca tək token üçün hesablamalar etdik. Bəzən söz birləşmələri, və ya ardıcıl tokenlər də önəmli məna daşıya bilər. Tekstdəki belə *n* ardıcıl tokendən ibarət birləşməyə ***n*-gram** deyilir. Bizim incələdiyimiz kimi tək tokenlər *unigram* adlanır. Daha böyük ölçülü birləşmələrin adlanması da *gram* ifadəsinin əvvəlinə uyğun ədədin latınca mənasını əlavə etməklə də aparılır (2 : bigram, 3 : trigram). Gəlin bir misal ilə n-gramları incələyək:

```
Dostum kitab oxumağı sevir. O, filmləri də sevir.
["dost", "kitab", "oxu", "sev", "film", "sev"]
```

- *1-gram* : ```'dost', 'sev'```, və s.
- *2-gram* : ```'kitab oxu', 'film sev'```, və s.
- *3-gram* : ```'dost kitab oxu'```, və s.

Digər n-gramlar üçün TF-IDF dəyərlərinin hesablanması unigramlar üçün etdiyimiz kimi aparılır. Adətən model qurularkən ixtiyari m dəyəri hesablamalarda arqument olaraq daxil edilir və lüğət $$\forall n, n 
\leq m$$ şərtinə uyğun olaraq n-gramlarla qurulur.

**Yekun qeydlər**

Bu yazıda incələdiyimiz TF-IDF modeli asan hesablanan və az resurs tələb edən modeldir. Hər hansı bir cümlədəki və ya mətndəki ən önəmli tokenləri TF-IDF dəyərlərinə uyğun olaraq tapmaq mümkündür. Bundan əlavə iki cümlənin/mətnin oxşarlığını onların TF-IDF vektorlarının arasındakı oxşarlıq əmsalından(məs. [kosinus əmsalı/oxşarlığı](https://en.wikipedia.org/wiki/Cosine_similarity)) istifadə edərək tapa bilərik. Bunlara baxmayaraq TF-IDF çanta(BoW) əsaslı model olduğuna görə tokenlərin cümlədəki/tekstdəki sırasını, semantik əlaqələri ehtiva etmir. Bu əskiklikləri aradan qaldırmaq üçün son illərdə müxtəlif vektorlaşdırma modelləri(word2vec, GloVe, fasttext, sent2vec, və s.) üzərində elmi araşdırmalar aparılır. Növbəti yazılar bu modellər haqqında olacaqdır.

Oxuduğunuz üçün təşəkkürlər!
