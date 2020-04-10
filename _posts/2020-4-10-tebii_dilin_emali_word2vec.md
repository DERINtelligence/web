---
layout: post
title: Təbii dilin emalı - word2vec.
author: Mammad Hajili
---

[Unitar kod](https://derintelligence.az/tebii_dilin_emali_giris/) sözlərin vektorizasiyası üçün önəmli üsuldur. Bu üsulda vektorlar mətndəki sözlərin verilən $$N$$ ölçülü lüğətdəki yerinə, yəni $$[0, N-1]$$ intervalında indekslərə uyğun dəyərlərin yazılması ilə əldə edilir. Sözün lüğətdəki yeri $$i$$-dirsə, həmin sözün $$N$$ ölçülü unitar kod vektorunda $$i$$ indeksinin dəyəri $$1$$, digər bütün indekslər $$0$$ olacaqdır. 

Bu üsul çox asan olsa da, bir çox halda əlverişli üsul deyildir. Ən önəmli problemlərdən biri unitar kod vektorlarlarının sözlər arasında bənzərliyi dəqiq ifadə edə bilməməsidir. Bu bənzərliyi ifadə etmək üçün istifadə olunan üsullardan biri kosinus oxşarlığıdır. Belə ki, iki vektor arasında bucağın kosinusu onlar arasında oxşarlığı ifadə edir. İxtiyari $$\mathbf{x}, \mathbf{y} \in \mathbb{R}^N$$ vektorları üçün olan kosinus bənzərliyi belə ifadə olunur: 

$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$

Əgər unitar vektorların xüsusiyyətinə nəzər salsaq, görə bilərik ki, iki müxtəlif sözün vektorları hər zaman perpendikulyardır ($$\mathbf{x}^\top \mathbf{y} =0$$). Deməli, iki müxtəlif sözün vektorları üçün kosinus oxşarlığı onların bir-birinə mənaca nə dərəcə yaxın və ya uzaq olmasından asılı olmayaraq hər zaman $$0$$-a bərabərdir. 

Sözlər arasında məna və əlaqəni dəqiq ifadə etməməsi unitar vektorların bir çox NLP tapşırıqlarda istifadəsini əlverişsiz edir. Bu yazıda haqqında danışacağım [Word2Vec](https://code.google.com/archive/p/word2vec/) bu problemi həll etmək üçün [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781) məqaləsində təklif edilmişdir. Bu üsulda hər bir söz təyin edilmiş ölçülü vektorla ifadə olunur və bu vektorlar sözlər arasında bənzərlikləri və əlaqələri daha geniş ehtiva edir. Word2Vec iki fərqli riyazi model təklif edir.

- Skip-gram
- Continuous bag of words (CBOW)

Biz yazının davamıda bu modellərə və onlar öyrətmə metodlarına baxacağıq.

**Skip-gram**

**Model**

Skip-gram modeldə token (mərkəzi söz) mətndə ətrafında olan tokenlərin (məzmun/qonşu sözlər) hansı sözlərə uyğun gəldiyini müəyyən etməkdə istifadə olunur. Gəlin bir nümunəyə nəzər yetirək:

    Mətn: "O", "oxuduğu", "kitab", "haqqında", "danışırdı"
    Mərkəzi söz: "kitab"
    Məzmun sözlər: "O", "oxuduğu", "haqqında", "danışırdı"

Bu nümunədə "kitab" sözünün mətndəki məzmun sözləri kimi onun 2 token məsafəsindəki sözlər qəbul edilmişdir (soldakı qonşular - "O", "oxuduğu", sağdakı qonşular - "haqqında", "danışırdı"). Bu məsafə dilçilikdə çərçivə (ing. window size) adlanır. 

Skip-gram modelində bu asılılıq şərti ehtimalla ifadə olunur, belə ki verilmiş mərkəzi söz üçün mətndəki qonşu sözlərin işlənmə ehtimalı belədir: $$P(\textrm{"O"},\textrm{"oxuduğu"},\textrm{"haqqında"},\textrm{"danışırdı"}\mid\textrm{"kitab"})$$  Verilmiş mərkəzi söz üçün məzmun sözlərin işlənməsi bir-birindən müstəqil hadisələr kimi qəbul edirik və buna görə də yuxarıdakı ifadəni belə ifadə bilərik:

$$P(\textrm{"O"}\mid\textrm{"kitab"})\cdot P(\textrm{"oxuduğu"}\mid\textrm{"kitab"})\cdot P(\textrm{"haqqında"}\mid\textrm{"kitab"})\cdot P(\textrm{"danışırdı"}\mid\textrm{"kitab"}).$$

<div class="center">
	<img src="https://i.imgur.com/QuSsa8q.png" style="width:50%;">
</div>

Bu ehtimalları hesablamaq üçün skip-gram modeldə hər bir söz $$d$$ ölçülü iki fərqli vektorla ifadə olunur, belə ki, verilən lüğətdə indeksi $$i$$ olan söz mərkəzi söz olduqda $$\mathbf{v}_i\in\mathbb{R}^d$$ vektoru ilə, məzmun söz olduqda isə, $$\mathbf{u}_i\in\mathbb{R}^d$$ vektoru ilə ifadə olunur. 

Gəlin, bizə lazım olan şərti ehtimalların necə ifadə edildiyinə baxaq. İxtiyari mərkəzi $$w_c$$ və məzmun $$w_o$$ sözləri və $$\mathcal{V}$$ lüğəti üçün məzmun sözün işlənməsinin mərkəzi sözdən asılılığını ifadə edən şərti ehtimal iki sözün vektorlarının vektoral hasilinin [softmax](https://en.wikipedia.org/wiki/Softmax_function) dəyəri ilə ifadə olunur. 

$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}$$

Verilən $$T$$ ölçülü mətn üçün yuxarıdakı ifadəni ümumiləşdirmək üçün mətndəki hər bir mərkəzi sözün məzmun sözlərlə olan şərti ehtimalının birgə ehtimalını hesablamalıyıq. Burada hər bir mərkəzi sözə uyğun şərti ehtimalların digər mərkəzi sözlərdən asılı olmadığını qəbul etsək ($$a$$ mərkəzi sözünə uyğun məzmun sözlərin $$a$$-dan asılılığı hadisəsi və $$b$$ sözü üçün eyni hadisə müstəqil hadisələrdir), birgə ehimalı aşağıdakı kimi ifadə edə bilərik: 

$$\prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)})$$

**Öyrətmə**

Digər maşın öyrənmə alqoritmlərində olduğu kimi bu modeldə də əsas məqsəd öyrətmə zamanı yuxarıda haqqında danışdığımız riyazi modelin nəticəsi olan birgə ehtimalın maksimallaşdırmasıdır. Bu anlayış öyrənmə nəzəriyyəsində *maksimal mümkünlük* adlanır. Bu mümkünlüyü öyrənmə alqoritmində işlədə bilmək üçün onu minimallaşdırma probleminə çevirməliyik. Loqarifmik funksiya monoton artan funksiya olduğu üçün, loqarifmik mümkünlük funksiyasının əksindən xəta funksiyası olaraq istifadə edə bilərik.

$$- \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)})$$

Xəta funsiyasının optimizasiyası üçün [stoxastik nöqtəvi meyilli azalma](https://derintelligence.az/suni_neyron_shebekeler_mashin_oyrenme/)  alqoritmini istifadə edə bilərik. Bunun üçün hər iterasiyada verilən mətndən təsadüfi kiçik hissə seçib, nöqtəvi meyilləri hesablayaraq modelin parametrlərini yeniləyəcəyik. Burada ən önəmli əməliyyat loqarifmik şərti ehtimalın nöqtəvi meylinin hesablanmasıdır. İlk olaraq şərti ehtimalı xatırlayaq

$$\log P(w_o \mid w_c) =
\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right)$$

Əgər yuxarıdakı ifadə üçün differensial tənliyi həll etsək, $$\mathbf{v}_c$$ üçün nöqtəvi meyil belə olacaqdır.

$$
\begin{split}\begin{aligned}
\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}
&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\
&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\
&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.
\end{aligned}\end{split}
$$

Öyrətmədən sonra verilən lüğətdəki ixtiyari $$i$$ indeksindəki söz üçün iki fərqli vektor - $$\mathbf{v}_i$$ və $$\mathbf{u}_i$$ əldə edirik. NLP tapşırıqlarında əsasən mərkəzi söz vektoru $$\mathbf{v}_i$$ istifadə olunur.

**Continuous Bag of Words(CBOW)**

**Model**

word2vec üsulunda təklif olunan digər model isə CBOW modelidir. Bu modelin skip-gram ilə oxşarlıqları olsa da, ondan çox fundamental bir xüsusiyyətlə fərqlənir. Belə ki, skip-gramın əksinə bu modeldə mərkəz sözün işlənməsi məzmun sözlərə əsaslanır. Bu halda şərti ehtimalı belə ifadə edirik.

$$P(\textrm{"kitab"}\mid \textrm{"O"}, \textrm{"oxuduğu"},\textrm{"haqqında"},\textrm{"danışırdı"})$$

<div class="center">
	<img src="https://i.imgur.com/2NQpYPu.png" style="width:50%;">
</div>

Çərçivəni $$m$$ qəbul etsək, verilən $$w_c$$ mərkəzi sözü üçün məzmun sözləri $$\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$$ ilə ifadə edə bilərik. Bu modeldə məzmun sözlərin sayı çox olduğundan softmax dəyəri hesablayarkən mərkəzi sözə uyğun gələn vektorla məzmun sözlərin vektorlarının ədədi ortasından istifadə edəcəyik. 

$$\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$$

Deməli, verilən mərkəzi söz və məzmun sözlər üçün şərti ehtimal aşağıdakı kimi olacaqdır.

$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$

və ya, qısaca:

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}$$

Yuxarıdakı ifadələrdən istifadə edərək modelin mümkünlük funksiyasını skip-gramda olduğu kimi müstəqil hadisələrin hasili ilə hesablamaq mümkündür:

$$\prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)})$$

**Öyrətmə**

CBOW modeli üçün öyrətmə metodu skip-gramdakı ilə demək olar ki, eynidir. Burada da yuxarıda hesabladığımız mümkünlük dəyərini maksimallaşdırmaq elə xəta funksiyasını minimallaşdırmağa ekvivalentdir. 

$$-\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)})$$

Yuxarıda qeyd etdiyimiz şərti ehtimaldan istifadə etsək, $$w_c$$ mərkəzi sözü üçün loqarifmik şərti ehtimal belə olacaqdır:

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right)$$

Bu ehtimala əsasən differensial tənliyi həll etsək, məzmun sözlərin ixtiyari biri $$v_{o_i}$$  üçün nöqtəvi meyli aşağıdakı kimi hesablaya bilərik.

$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$

Skip-gram modelindən fərqli olaraq NLP tapşırıqlarda əsasən bu modelin məzmun söz vektorlarından istifadə olunur.

**Nəticələr və yekun**

word2vec modellərində öyrənmə zamanı yuxarıda da danışdığımız kimi yalnızca mətndən istifadə edilir, yəni verilən data işarələnməmişdir. Yəni sözlər arasında əlaqələr, məna yaxınlığı və ya uzaqlığı haqqında əvvəldən heç bir məlumata sahib olmuruq. Bununla belə, word2vec olduqca aydın və maraqlı nəticələr göstərir. Məsələn, elə müəlliflərin mövzu ilə əlaqəli digər məqaləsindən ([Mikolov et al. 2013](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) ) ingilis dilli mətn əsasında yaradılmış 1000 ölçülü word2vec vektorlarının PCA alqoritmi ilə 2 ölçülü müstəvidə proyeksiyasına nəzər yetirək:

<div class="center">
	<img src="https://i.imgur.com/VOm2SEC.jpg" style="width:70%;">
</div>

Aydın şəkildə görə bilərik ki, "dövlət-paytaxt" əlaqəsi ölkə və şəhər vektorları arasında, demək olar ki, eynidir.

Eyni məqalədən digər bir misala nəzər salaq. Aşağıdakı cədvəl iki sözün vektorlarının cəminə ən çox yaxın olan vektorların hansı sözlərə uyğun gəldiyini göstərir.

<div class="center">
	<img src="https://i.imgur.com/w3kRSHn.jpg" style="width:70%;">
</div>

Bu misal ilə vektorlar üzərində edilən riyazi əməliyyatların necə effektiv olduğunu və modelin vektorların oxşarlıq əlaqələrini necə ehtiva etdiyini aydın görə bilərik.

word2vec vektorizasiya üsulunun digər üstün cəhətlərindən biri də böyük korpus əsasında öyrədilmiş vektorları $${söz : vektor}$$ formatında ikililərdən ibarət lüğət olaraq saxlayıb fərqli NLP tapşırıqlar üçün istifadə edə bilməyimizdir. Belə lüğətlərə NLP-də **embedinqlər** deyilir. 


Embedinqlərin istifadəsi müasir NLP-nin və hal-hazırda ən önəmli maşın öyrənmə mövzularından olan transfer öyrənmənin əsasını təşkil edir. word2vec müasir embedinqlərin ilk nümunəsi hesab edilir və olduqca önəmli elmi araşdırma mövzularına ilham vermişdir. Hal-hazırda dərin öyrənmənin də sürətli inkişafı ilə çox effektiv nəticələr verən embedinqlər istifadə edilir. Növbəti mövzularda onlar haqqında danışmağa çalışacağam. 

Oxuduğunuz üçün təşəkkürlər, salamat qalın!

**Ədəbiyyat**

- [Efficient Estimation of Word Representations in Vector Space](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1301.3781)
- [14.1. Word Embedding (word2vec), d2l.ai](http://d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html#why-not-use-one-hot-vectors)
