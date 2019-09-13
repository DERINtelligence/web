---
layout: post
title: Guess who's back'propagate. - Xəta funksiyası və stoxastik nöqtəvi meyilli azalma.
---
<style type="text/css">
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;Bizim
}
</style>

Salamlar, keçən [yazıda](http://derintelligence.az/suni_neyron_shebekeler_ireli_qidalama/) süni neyron şəbəkənin arxitekturası haqqında danışdıq. Orada göstərdiyim "irəli qidalama" tərcüməsi ilə bağlı bir neçə mesaj aldım və biraz qəribə tərcümə olduğunu nəzərə alaraq onu "irəliyə ötürmə" kimi dəyişməyin daha düzgün olduğunu düşünürəm. Bundan əvvəlki yazıda da uyğun dəyişiklikləri edəcəyəm.

Bu dəfəki yazımızda maşın öyrənmənin ümumi izahından, istifadə olunan bərabərliklərdən və alqoritmlərdən - xəta funksiyasından və stoxastik nöqtəvi meyilli azalma haqqında danışacam. Bu yazını yazarkən, oxuyucumun ən azı maşın öyrənmənin nə olduğunu bildiyini ümid edirəm. Hazır olun, bu yazıda bol-bol riyazi ifadələr görəcəksiniz. Kəmərləri bağlayın, başlayırıq.

<img src="https://raw.githubusercontent.com/DERINtelligence/web/master/images/paris_blog.jpg" style="width:700px;height:400px">

Əslində bu yazını yazmaq bir qədər çətindir, çünki, maşın öyrənmə alqoritmlərini tam izah etmək üçün 1 yox, 5 yazı bəs etməz, böyük ehtimalla. Ona görə də mən bu yazıda yalnızca bizim SNŞ-dəki hesablamalar üçün lazım olan önəmli nüanslara toxunacam. Belə ki, maşın öyrənmə alqoritmləri(biz burada öyrənməni müşahidəli(ing. supervised) olaraq hesab edirik) verilənlər toplusu $$X={X_1, X_2, \cdots, X_N}$$ və ona uyğun cavablar $$Y={Y_1, Y_2, \cdots, Y_N}$$ əsasında qeyri-xətti və ya xətti bir qanunauyğunluq taparaq verilənlər toplusu ilə eyni quruluşda olan yeni verilənə uyğun cavabı təxmin etmək üçün istifadə edilir. Bu iki ardıcıl prosesə "öyrən və təxmin et" (ing. learn and predict) də deyə bilərik. Ancaq bu prosesdə, məncə bir nüans əskikdi, sizcə, hansı? 

$$N$$ ölçülü bu topluda ən optimistik halda biz hər bir elementin yalnızca uyğun cavab dəyəri ilə uyğunluğunu görə bilərik. Ancaq toplunun hər verilənin uyğunluğu individual olaraq bizim məsələdə bir məna kəsb etmir, çünki ixtiyari iki verilən - $$X_i$$ və $$X_j$$ fərqli-fərqli funksiyanın təyin oblastı ola bilər. Bizə ümumi qanunauyğunluğu tapmaq üçün bütün bu funksiyaları ehtiva edəcək ümumi funksiyanı - riyazi modeli tapmaq gərəklidir. Gəlin, bu funksiyanı $$f(X)$$ ilə işarə edək. Ən sadə halda bu funksiya arqumentləri girişə uyğun ağırlıq vektoru və sürüşmə əmsalı olan xətti funksiya ola bilər. Biz də hesablamalar zamanı izahı daha asan olsun deyə bu funksiyadan istifadə edəcəyik.

\begin{eqnarray} 
		f(X_i) = w \cdot X_i + b
\tag{1}\end{eqnarray} 

Yuxarıda qeyd etdiyim verilənlər toplusunda gördüyünüz kimi hər bir element $$X_i$$-in ona uyğun cavabı $$Y_i$$ var. Deməli biz ümumi funksiyamızın təxmininin dəqiqliyini funksiyanın dəyərinin və verilən elementə uyğun cavabın bir-birinə nə qədər yaxın olduğu ilə müqayisə edə bilərik. Bu yaxınlığı/uzaqlığı hesablamaq üçün **xəta funksiyasından**(ing. loss/cost function) istifadə edirik. Bu məsələmizdə xəta funksiyası kimi kvadratik xəta funksiyasında istifadə edəcəyik. $$f(X)$$ funksiyasının ümumi xətası bütün elementlərə uyğun xətaların cəmi olduğundan kvadratik funksiya cəm zamanı mənfi və müsbət xətaların bir-birini ixtisar etməsinin qarşısını alır. Gəlin $$f(X)$$ funksiyasının xəta funksiyasını $$L(X)$$ ilə işarə edək. Cəmi normallaşdırma üçün xətalar cəmini nöqtələrin sayına bölürük.

\begin{eqnarray} 
		L(X_i) = \frac{1}{2}{(Y_i - f(X_i))} ^ 2
\tag{2}\end{eqnarray} 

\begin{eqnarray} 
		L(X) = \frac{1}{N}\sum_i L(X_i)
\tag{3}\end{eqnarray} 

**Qeyd:** Kvadratik funksiyanı $$\frac{1}{2}$$ əmsalı ilə vurmağımızın səbəbi sonrakı hesablamalar zamanı törəmə aldığımızda qalıq əmsalsız nəticə əldə etməkdir. Bu ümumi qəbul edilmiş bir qaydadır, ancaq, təbii ki, bütün hesablamaları bu əmsalı nəzərə almadan da etsək, yenə də eyni nəticəyə gələcəyik.

Xəta funksiyası maşın öyrənmə alqoritmlərinin təməlini təşkil edir, belə ki, yuxarıda qeyd etdiyimiz əskik nüansı onu istifadə edərək tamamlaya bilərik. Bər. 1-də gördüyümüz kimi funksiyanın dəyəri ağırlıq və sürüşmə əmsalından asılıdır və Bər. 2-də gördüyümüz kimi xətanın dəyəri $$f(X)$$-in dəyərindən asılıdır. Yəni, bilavasitə, modelin xətası $$f(X)$$ funksiyasının arqumentləri olan ağırlıq $$w$$ və sürüşmə $$b$$ dəyərlərindən asılıdır. Bizim məqsədimiz isə, ən dəqiq, yəni verilən cavaba ən yaxın dəyərlər təxmin etməkdir, başqa bir sözlə desək xətanı azaltmaqdır. Deməli, biz funksiyanın arqumentlərini dəyişməklə ən az xətalı modeli əldə edə bilərik. Yəni, ixtiyari təyin olunmuş $$w^{(0)}$$ və $$b^{(0)}$$-dən başlayaraq, xəta azalana qədər onları *müəyyən üsulla* dəyişsək modelimiz öyrənmə prosesini bitirəcək. Ona görə də, mən təklif edirəm ki, bu iterativ məntiqi də nəzərə alaraq, gəlin, "öyrən və təxmin et" prosesini "öyrədilənə qədər davam et və təxmin et" olaraq dəyişək və əskik nüansı aradan qaldıraq.

Yuxarıdakı abzasda sondan ikinci cümlədə fikir versəniz "müəyyən üsulla" hissəsini xüsusi işarələdim. İndi bizi maraqlandıran isə bizə ən dəqiq modeli verəcək ən optimal $$w$$ və $$b$$-ni hər iterasiyada hansı üsulu istifadə edərək dəyişəcəyimizi müəyyən etməkdir. Maşın öyrənmədə, təbii ki, bir-birindən fərqli müxtəlif üsullar təklif olunub, ancaq mən sizə SNŞ-lərin döyünən ürəyi olan **stoxastik nöqtəvi meyilli azalmanı(SNMA)**(ing. stochastic gradient descent(SGD)) göstərəcəm. Bu arada, bu mövzu hələ də üzərində kifayət qədər elmi araşdırma gedən mövzudur və dərin riyazi izahı vaxt alandır, ona görə də mən riyazi analizin axtarışını sizin ixtiyarınıza buraxıram. Bununla belə, bizə lazım olacaq riyazi izahı verəcəm, təbii ki. 

<img src="https://raw.githubusercontent.com/DERINtelligence/jekyll-now/master/images/sgd.png" style="width:700px;height:500px">

Belə ki, **nöqtəvi meyil(ing. gradient)** funksiyanın toxunanının müəyyən nöqtədəki meylidir. Nöqtəvi meyil, xəta funksiyasının artış istiqamətinin əksinə olan vektor olaraq da ifadə edilir. Ancaq biz burda qarışıqlıq olmasın deyə onun ədədi dəyəri ilə ifadə edəcəyik. Belə ki, ədədi dəyər ilə ifadə edildiyində nöqtəvi meyil, sadəcə xəta funksiyasının müəyyən nöqtədəki differensialının əksinə bərabər olur. $$L$$ funksiyası $$w$$ və $$b$$-dən asılı funksiya olduğundan, iki nöqtəvi meyildən istifadə edəcəyik. Belə ki, hər hansı bir arqumentin $$t+1$$ anındakı dəyərini tapmaq üçün onun $$t$$ anındakı dəyəri ilə həmin andakı nöqtəvi meylini toplayacağıq. 

\begin{eqnarray} 
		w^{(t+1)} = w^{(t)} - \beta_1\frac{\partial L_n}{\partial w^{(t)}}
\tag{4}\end{eqnarray}

\begin{eqnarray} 
		b^{(t+1)} = b^{(t)} - \beta_2\frac{\partial L_n}{\partial b^{(t)}}
\tag{5}\end{eqnarray}

Burada $$\beta_1$$ və $$\beta_2$$ xəta dəyərinin çox böyük qiymətlərində $$w$$ və $$b$$-nin kəskin şəkildə azalmasının qarşısını alaraq modelin normallaşdırılması üçün istifadə olunan əmsallardır. $$L_n$$ isə verilənlər toplusundan ixtiyari seçilmiş $$n$$ elementinin xəta dəyəridir. Bu element hər iterasiyada ixtiyari olaraq seçildiyindən bu alqoritmi **stoxastik** nöqtəvi meyilli azalma adlandırırıq.

**Əlavə tapşırıq:** Bər. 2-də $$L$$ funksiyasını görürük. Bu bərabərlik əsasında törəmələri həll edərək Bər 4. və Bər 5.-i yenidən göstərin.

Bir daha xatırlatmaq istəyirəm ki, $$f(X)$$ funksiyasını izahı sadə olduğundan istifadə etdim, növbəti yazıda indi öyrəndiyimiz alqoritmlərin SNŞ-lərə tətbiqi zamanı $$f(X)$$ yox, gizli qatların aktivasiyası və yekun çıxış qatının dəyərindən istifadə edəcəyik. Açığı, burada xəta funksiyası və SNMA-nı yaxşı anlasanız, növbəti yazı sizin üçün yalnızca mürəkkəb funksiyaların törəməsinin **zəncir qaydası** ilə həllini başa düşməkdən ibarət olacaq. Bəli, gələn yazıda seriyanın adını daşıyan geriyə yayılma(backpropagation) alqoritmindən danışacağıq. 

Buraya qədər səbr edib oxuduğunuz üçün təşəkkürlər!
